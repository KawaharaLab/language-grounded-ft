/**
 * Low-latency MMS101 force/torque sensor driver for ROS 2.
 *
 * The sensor streams one 25-byte frame at the configured interval. Serial
 * reception is handled by a blocking worker thread instead of a ROS timer:
 * timer polling can drift relative to the sensor and partial reads otherwise
 * leave the stream permanently out of frame.
 */

#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cerrno>
#include <chrono>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

#ifdef __linux__
#include <linux/serial.h>
#include <sys/ioctl.h>
#endif

class MMS101Node : public rclcpp::Node
{
public:
  MMS101Node()
  : Node("mms101_driver")
  {
    this->declare_parameter("port", "/dev/ttyUSB1");
    this->declare_parameter("frame_id", "sensor_link");
    this->declare_parameter("force_scale", 0.001);
    this->declare_parameter("torque_scale", 0.00001);
    this->declare_parameter("sample_interval_us", 10000);
    this->declare_parameter("low_latency", true);

    port_name_ = this->get_parameter("port").as_string();
    frame_id_ = this->get_parameter("frame_id").as_string();
    force_scale_ = this->get_parameter("force_scale").as_double();
    torque_scale_ = this->get_parameter("torque_scale").as_double();
    sample_interval_us_ = this->get_parameter("sample_interval_us").as_int();
    low_latency_ = this->get_parameter("low_latency").as_bool();

    if (sample_interval_us_ < 1000 || sample_interval_us_ > 0xFFFF) {
      throw std::invalid_argument("sample_interval_us must be in [1000, 65535]");
    }

    // Keep only the freshest sample while retaining reliable compatibility
    // with the existing capture and inference subscribers in this workspace.
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
    qos.reliable();
    qos.durability_volatile();
    publisher_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>(
      "force_torque", qos);

    if (!open_serial_port()) {
      throw std::runtime_error("Failed to open MMS101 serial port " + port_name_);
    }
    if (!initialize_sensor()) {
      close(serial_fd_);
      serial_fd_ = -1;
      throw std::runtime_error("Failed to initialize MMS101 on " + port_name_);
    }
    if (!configure_streaming_read()) {
      close(serial_fd_);
      serial_fd_ = -1;
      throw std::runtime_error("Failed to configure MMS101 streaming read");
    }

    running_ = true;
    reader_thread_ = std::thread(&MMS101Node::reader_loop, this);
    RCLCPP_INFO(
      this->get_logger(),
      "MMS101 streaming on %s every %ld us (target %.1f Hz)",
      port_name_.c_str(), sample_interval_us_, 1.0e6 / sample_interval_us_);
  }

  ~MMS101Node() override
  {
    running_ = false;
    if (reader_thread_.joinable()) {
      reader_thread_.join();
    }
    stop_measurement();
    if (serial_fd_ != -1) {
      close(serial_fd_);
      serial_fd_ = -1;
    }
  }

private:
  static constexpr std::size_t kFrameSize = 25;

  int serial_fd_{-1};
  std::string port_name_;
  std::string frame_id_;
  double force_scale_{0.001};
  double torque_scale_{0.00001};
  int64_t sample_interval_us_{10000};
  bool low_latency_{true};
  std::atomic_bool running_{false};
  uint64_t frame_count_{0};
  uint64_t dropped_byte_count_{0};
  std::thread reader_thread_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr publisher_;

  bool open_serial_port()
  {
    // O_SYNC affects writes and can add unnecessary scheduling delay here.
    serial_fd_ = open(port_name_.c_str(), O_RDWR | O_NOCTTY);
    if (serial_fd_ < 0) {
      RCLCPP_ERROR(
        this->get_logger(), "Error opening %s: %s", port_name_.c_str(), strerror(errno));
      return false;
    }

    struct termios tty {};
    if (tcgetattr(serial_fd_, &tty) != 0) {
      RCLCPP_ERROR(this->get_logger(), "tcgetattr failed: %s", strerror(errno));
      return false;
    }

    cfmakeraw(&tty);
    cfsetospeed(&tty, B1000000);
    cfsetispeed(&tty, B1000000);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag |= CLOCAL | CREAD;
    tty.c_cflag &= ~(PARENB | PARODD | CSTOPB | CRTSCTS);
    // Initialization reads are allowed to time out after 100 ms.
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;

    if (tcsetattr(serial_fd_, TCSANOW, &tty) != 0) {
      RCLCPP_ERROR(this->get_logger(), "tcsetattr failed: %s", strerror(errno));
      return false;
    }
    tcflush(serial_fd_, TCIOFLUSH);
    enable_low_latency_best_effort();
    return true;
  }

  void enable_low_latency_best_effort()
  {
#ifdef __linux__
    if (!low_latency_) {
      return;
    }
    struct serial_struct serial_info {};
    if (ioctl(serial_fd_, TIOCGSERIAL, &serial_info) != 0) {
      RCLCPP_DEBUG(
        this->get_logger(), "TIOCGSERIAL is not supported on %s: %s",
        port_name_.c_str(), strerror(errno));
      return;
    }
    if ((serial_info.flags & ASYNC_LOW_LATENCY) != 0) {
      return;
    }
    serial_info.flags |= ASYNC_LOW_LATENCY;
    if (ioctl(serial_fd_, TIOCSSERIAL, &serial_info) != 0) {
      RCLCPP_WARN(
        this->get_logger(),
        "Could not enable ASYNC_LOW_LATENCY on %s (continuing): %s",
        port_name_.c_str(), strerror(errno));
    } else {
      RCLCPP_INFO(this->get_logger(), "Enabled ASYNC_LOW_LATENCY on %s", port_name_.c_str());
    }
#endif
  }

  bool configure_streaming_read()
  {
    struct termios tty {};
    if (tcgetattr(serial_fd_, &tty) != 0) {
      return false;
    }
    // Return as soon as data arrives, with a 100 ms no-data timeout so the
    // worker can always observe shutdown even if the sensor is unplugged.
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;
    return tcsetattr(serial_fd_, TCSANOW, &tty) == 0;
  }

  bool write_all(const std::vector<uint8_t> & command)
  {
    std::size_t offset = 0;
    while (offset < command.size()) {
      const ssize_t written = write(
        serial_fd_, command.data() + offset, command.size() - offset);
      if (written > 0) {
        offset += static_cast<std::size_t>(written);
      } else if (written < 0 && errno == EINTR) {
        continue;
      } else {
        RCLCPP_ERROR(this->get_logger(), "Serial write failed: %s", strerror(errno));
        return false;
      }
    }
    tcdrain(serial_fd_);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    return true;
  }

  void flush_response()
  {
    uint8_t buffer[128];
    (void)read(serial_fd_, buffer, sizeof(buffer));
  }

  bool initialize_sensor()
  {
    RCLCPP_INFO(this->get_logger(), "Initializing MMS101 on %s", port_name_.c_str());

    if (!write_all({0x54, 0x02, 0x10, 0x00})) {return false;}
    flush_response();

    if (!write_all({0x54, 0x03, 0x36, 0x00, 0x01})) {return false;}
    flush_response();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    if (!write_all({0x54, 0x03, 0x36, 0x05, 0x01})) {return false;}
    flush_response();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    for (uint8_t axis = 0; axis < 6; ++axis) {
      if (!write_all({0x54, 0x02, 0x1C, axis})) {return false;}
      flush_response();
      if (!write_all({0x53, 0x02, 0x57, 0x94})) {return false;}
      flush_response();
      std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }

    if (!write_all({0x54, 0x01, 0xB0})) {return false;}
    flush_response();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const auto interval = static_cast<uint16_t>(sample_interval_us_);
    if (!write_all(
    {
      0x54, 0x04, 0x43, 0x00,
      static_cast<uint8_t>((interval >> 8) & 0xFF),
      static_cast<uint8_t>(interval & 0xFF)}))
    {
      return false;
    }
    flush_response();

    if (!write_all({0x54, 0x02, 0x23, 0x00})) {return false;}
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    flush_response();
    return true;
  }

  void stop_measurement()
  {
    if (serial_fd_ != -1) {
      (void)write_all({0x54, 0x01, 0x33});
    }
  }

  static int32_t parse_24bit_be(const uint8_t * data)
  {
    int32_t value =
      (static_cast<int32_t>(data[0]) << 16) |
      (static_cast<int32_t>(data[1]) << 8) |
      static_cast<int32_t>(data[2]);
    if ((value & 0x800000) != 0) {
      value |= static_cast<int32_t>(0xFF000000);
    }
    return value;
  }

  void reader_loop()
  {
    std::vector<uint8_t> receive_buffer;
    receive_buffer.reserve(512);
    uint8_t chunk[256];

    while (running_ && rclcpp::ok()) {
      const ssize_t count = read(serial_fd_, chunk, sizeof(chunk));
      if (count > 0) {
        receive_buffer.insert(receive_buffer.end(), chunk, chunk + count);
        parse_receive_buffer(receive_buffer);
      } else if (count < 0 && errno != EINTR && errno != EAGAIN) {
        RCLCPP_ERROR_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Serial read failed on %s: %s", port_name_.c_str(), strerror(errno));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

  void parse_receive_buffer(std::vector<uint8_t> & buffer)
  {
    while (buffer.size() >= 4) {
      std::size_t start = 0;
      while (
        start + 3 < buffer.size() &&
        !(buffer[start] == 0x00 && buffer[start + 1] == 0x17 &&
        buffer[start + 2] == 0x80 && buffer[start + 3] == 0x00))
      {
        ++start;
      }
      if (start > 0) {
        dropped_byte_count_ += start;
        buffer.erase(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(start));
      }
      if (buffer.size() < kFrameSize) {
        return;
      }

      publish_frame(buffer.data());
      buffer.erase(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(kFrameSize));
    }
  }

  void publish_frame(const uint8_t * frame)
  {
    geometry_msgs::msg::WrenchStamped message;
    // Closest host-clock approximation to the sensor sample time: immediately
    // after the complete 25-byte frame is available.
    message.header.stamp = this->now();
    message.header.frame_id = frame_id_;
    message.wrench.force.x = parse_24bit_be(&frame[4]) * force_scale_;
    message.wrench.force.y = parse_24bit_be(&frame[7]) * force_scale_;
    message.wrench.force.z = parse_24bit_be(&frame[10]) * force_scale_;
    message.wrench.torque.x = parse_24bit_be(&frame[13]) * torque_scale_;
    message.wrench.torque.y = parse_24bit_be(&frame[16]) * torque_scale_;
    message.wrench.torque.z = parse_24bit_be(&frame[19]) * torque_scale_;
    publisher_->publish(message);

    ++frame_count_;
    if (frame_count_ % 1000 == 0) {
      RCLCPP_INFO(
        this->get_logger(), "Published %lu frames; discarded %lu framing bytes",
        static_cast<unsigned long>(frame_count_),
        static_cast<unsigned long>(dropped_byte_count_));
    }
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<MMS101Node>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("mms101_driver"), "%s", error.what());
  }
  rclcpp::shutdown();
  return 0;
}
