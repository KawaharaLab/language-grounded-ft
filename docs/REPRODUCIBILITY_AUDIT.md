# 論文原稿：再現性監査と appendix 改稿ノート

監査・改稿日：2026-09-14（JST）。対象：ローカルの原稿（ソース非同梱）。参照：[EXPERIMENT_PATHS.md](EXPERIMENT_PATHS.md)。

## 1. 今回の結論と変更範囲

要求された6項目について、実際に学習へ渡したCSV、保存metadata、checkpointの必要な設定、実験時のコードsnapshot、実機の試行設定・集計を照合した。本文の確定した誤記を修正し、既存appendixの一般的な説明を、観測量の式・ラベル定義・分布・分割・計算条件へ置き換えた。

**実験結果を確定するには、支持面分類31/40の元スコアと短時間wobble Fz評価0/20の採用試行を確定する必要がある。** 保存場所について「すべてur3_robotiq_ros2、木片はrectangle」との著者補足を受けて追加探索したが、この2点はまだ特定できていない。数値が誤りと確定したわけではないため元の値を保持し、表のcaption・記号とappendix末尾に可視の著者確認注記を付けた。未確認値を確認済みとして扱わない。

本文・appendixの修正だけで解消できない点として、GSO validationの時間的重複と、仮想・実機トルク原点の厳密な対応がある。前者は既存実験の事実として明記し、後者について原点移動を実施したとの主張をしない。収集・学習・元データ・既存評価結果は変更していない。再学習や全モデルの再推論も行っていない。

計算機は著者回答で **NVIDIA RTX A6000×2**。生成ログのデバイス表示とも一致する。断続的な追加収集だったため、全データ生成時間や総GPU-hoursを推測して記載しない。区間の揃った一部収集ログだけを限定した例として示す。

### 根拠の区別

- **保存確認**：対象版CSV・metadata・checkpoint・試行JSON・当時のコードsnapshotから確認した事実。
- **再集計**：保存データを数え直した値。splitについては保存seedと分割処理からの再構成を含む。
- **実装確認**：現checkoutから読み取れる処理。過去全実験と同一のコード・依存環境だったとまでは断言しない。
- **未特定**：原票・設定・実行履歴を回収できず、現存資料だけでは確定できないもの。

### 本文に反映した主な訂正

| 元の記述・不足 | 対応 |
| --- | --- |
| “held-out GSO subset” | window単位90/10で物体・episodeを分離しないことを明記。seed42のvalidation 39,646/43,724窓（90.67%）がtrainと生サンプルを共有する再集計もappendixに記載。YCBの物体分離とは区別。 |
| 仮想センサの定義・トルク原点が不明 | 各指の全接触力合計、指link原点まわりのモーメント、指local座標の式を追加。実機側はRz(pi)のみで原点移動なしと明記。 |
| 全windowが0.1秒stride | 通常strideと終端accidental-drop用windowの例外を明記。10点grid外は14,383窓。 |
| 力100 N・移動5 cmという曖昧な除外条件 | 6力成分の絶対値>=100 N、COMの1step変位>0.05 mと不等号・対象量を明示。 |
| weight/slipが一般的な言葉のみ | light<0.2 kg / heavy>=0.2 kg、接触による欠損規則、5点連続接触、最大接線変位/.79 sの式を追加。 |
| Sentence-BERTのモデル名なし | sentence-transformers/all-mpnet-base-v2、768次元、L2正規化、固定温度.07を記載。 |
| 初期固定面も毎把持前1秒のオフセット補正 | 初期固定面encoder60試行は補正なし、wobbleとremountは補正ありへ訂正。Fz baselineのmedian tareは別処理と明記。参照元pathsの該当箇所も訂正。 |
| RQ2のCNN/MLP結果でarchitecture選択を正当化 | センサ再取付け・校正・補正の違う別sessionであることを明示し、architectureだけの比較とはしない。 |
| Fzは厳密な“crossing”検知 | 各指の把持前1秒medianを引いた和<0、2秒gate後、50 ms毎の3回連続。前値が正であることは実装条件ではないと精密化。 |
| 0.300秒が“separately measured” | 補正適用は確認したが独立測定資料は未特定のため、固定reaction-time correctionという表現へ。 |
| コルクの19/20と保存集計19/19 | Enterなし早期検知失敗1件をattempt分母に残す規則を追記。平均delayは成功19件。 |
| RQ3手法列挙に教師ありがない | 図に含まれるsupervised multi-task pretrainingを追加。元headを外すこと、全encoderはpretrain seed42で5 seedsは下流の反復と明記。 |
| LoRAの適用層が曖昧 | 時間3層＋fusion1層のQ/V計8 maps、rank4・alpha8・8,961 parametersを明記。 |
| SSLにもLoRAが一般的に有利との広い記述 | VAE/VICRegは多くのNでfrozenが高いことを明示。BYOLにも例外があり“most labeled-data budgets”と限定。 |
| 綴り・引用符・保存文章との不一致 | 重複語、欠けた空白、引用符を修正。例文を保存templateの表記へ合わせた。 |

## 2. 要求された6項目をどこに載せるか

| 項目 | 原稿appendix | 詳細資料・残課題 |
| --- | --- | --- |
| 仮想F/Tとトルク基準点 | `app:wrench`：合成式、作用力の符号、link原点、左右local frame、実機の符号変換・原点移動なし | 本MD第3節。公開時に左右の取付寸法図・センサ測定原点を追加すると、幾何学的対応を判断できる。 |
| light/heavy・contact・slip | `app:annotation`：数値と演算順序、drop例外、weight欠損、supportラベル | 本MD第3節。現実装と76保存窓の照合結果、historical fallbackも記録。 |
| 記述の種類・頻度・欠損・遷移 | `app:sampling`：269→250文章、top10占有率、属性分布、mandatory保持 | 本MD第4節に前後分布・joint missingness、末尾に全269文章の前後頻度。 |
| GSO train/validation分割 | `app:split`：window単位、seed、件数、時間共有率 | 本MD第4節に3 seeds×対照/教師ありの全数監査。`paper_audit`に再集計コード。 |
| Sentence-BERTの具体モデル | 本文と`app:split`：モデルID、次元、正規化、文prototype | 本MD第4節。過去model revisionとlibrary版の厳密確認は残るため、保存embedding/index mapを正本として公開。 |
| 生成時間・計算資源・除外率 | `app:acquisition`：A6000×2、総時間不明の理由、限定campaign時間、GSO/YCB attrition | 本MD第5節。全GPU-hours、段階ごとの全worker失敗率や理由別window除外率は未記録部分あり。 |

今回のappendixは情報を落とさず確認できる詳細案である。ページ数制約への圧縮は組版環境を復旧した後に行い、必要なら詳細な分布・実験別例外をsupplementaryへ移す。本文の方法説明を繰り返すためではなく、各数値・ラベル・観測量を再構築するために内容を置いた。

## 3. 仮想F/T・教師ラベル・シミュレーション設定の根拠

コードの相対パスはGenesis基準。engineの力・トルク集計部分は調査時にGit差分なし。生成・annotationコードには既存の未コミット変更があるため、以下では保存版の値と現ソースからの復元を区別する。

### 仮想F/Tの定義

#### 保存された観測量を生む処理

`src/generation/make_step.py:125–137` は `scene.step()` 後に、Pandaの左右finger link（local index9,10）の `get_links_contact_force([9,10], sensor=True)` と `get_links_contact_torque(...,sensor=True)` を読む。

- 6軸の順序は各指 `[Fx,Fy,Fz,Tx,Ty,Tz]`、左→右の12軸。`run_genesis_sim.py:1643–1657` でもCSV列名確認。
- 単位は力N、トルクN m。API名にsensorとあるが、MMS101の専用センサ物理モデルではない。
- `genesis/engine/entities/rigid_entity/rigid_entity.py:4499–4525` はworld座標で蓄積された力/トルクを、各link quaternionの逆回転により、そのlinkのlocal座標へ変換する。
- `genesis/engine/solvers/rigid/constraint/solver.py:7039–7100` はステップ毎に各linkの接触力/トルクをゼロ化し、collision contactのconstraint forceを全接触点について加算する。接触対a,bに同じ符号で足すのではなく、aには `-force`、bには `+force`。これは「その指へ作用する力」。物体へ作用する力は反対符号。
- 同solverの7091–7100行で `torque = (contact_pos − dyn_state.links.pos[link]) × force_on_link`。基準点は**finger linkの原点**。物体COM、接触重心、手首原点ではない。

指iのworld原点をo_i、local→world回転をR_i、その指に作用するcontact jのworld力f_ij、接触world位置をp_ijとすると、

`F_i = R_i^T sum_j f_ij`

`tau_i = R_i^T sum_j ((p_ij - o_i) cross f_ij)`

加算対象はその指に直接作用する全collision接触であり、指–物体の接触だけに限定するfilterはない。法線反力と接線摩擦力を含む。現設定のtorsional/rolling frictionは無効（下記）。独立したfree torque、指自重/慣性の解析的付加、関節反力からのwrench、MMS101のノイズ/transfer functionはこの読み出しにはない。把持物体の重力・慣性・環境反力は、物体–指接触を介してこれらの力に影響する。

#### 原点と左右座標のasset根拠

実際に読むassetは `genesis/assets/xml/franka_emika_panda/panda.xml`（`run_genesis_sim.py:535–539`）であり、別URDFを引用しない。

- XML218–229行: `left_finger` bodyの基準posはhandから `[0,0,0.0584]` m、prismatic joint。
- XML230–241行: `right_finger`は同基準pos、quat `0 0 0 1`（wxyzでz180°回転）。実際の原点はjoint変位も含む毎stepのlink pose。
- 双方のinertial posはこのassetでは0だが、計算式の根拠はあくまでlink originであり「一般にCOM基準」と書くべきではない。

#### 実機との対応で断言できる範囲

`/home/user/ur3_robotiq_ros2/data_processing/compute_pick_place_force_normalization.py:33–40` はMMS101 rawからsim localへの設定を `[-1,-1,+1,-1,-1,+1]` ×左右、すなわちFとτに同じRz(pi)を適用と明記。座標・offset・scaleの対応は確認できる。一方、確認した実機前処理には、実機センサ原点からPanda link原点への `r cross F` トルク移動を見つけていない。実機センサの物理的測定中心とlink原点が一致するとの根拠は未特定。

従って「対応付ける量は指に伝わる6軸resultant/contact wrench」であり、「同じ原点で同じ完全なwrenchを測っている」「実機の指自重/慣性までsimulation sensorが厳密に再現」などは主張しない。必要な追加資料: MMS101の取り付け寸法/測定原点、左右の座標図、トルク原点差の扱い。

**RQ2の履歴差（第6節で全試行設定を確認済み）:** `/home/user/ur3_robotiq_ros2/artifacts/cnn_place_input_audit_20260910/README.md` は「8/27午後の元の固定面20件は空荷補正導入前」と記録している。初期固定面encoder60試行に補正はなく、後続wobble等では有効だった。本文をこの履歴に合わせて訂正した。

### ラベルを厳密に再現する定義

以下の主実装は `src/processing/rigid/1_create_annotations.py`。閾値5samples、slip .005m/sは対象版metadataでも保存確認済み。

#### windowと接触の確認

- 通常80samples、step間隔 .01s、start stride10samples。window nominal lengthは.8sだが**最初–最後のsample elapsed timeは.79s**（503–511行）。
- early accidental dropで通常grid windowが終端eventを含まない場合は `start=len(episode)-80` を追加（182–202行）。**balanced CSV全件を集計するとoff-grid startは14,383件、全件interaction=accidental drop**。従って一律stride0.1sと断言せず、終端event用の例外をappendixに書く。
- windowそのものの採用条件 `_has_bimanual_contact`（105–118行）は「window内で左指接触が1回以上、かつ右指接触が1回以上」。同時接触が必須ではない。最終CSVに同時接触0点の採用窓が408件ある（place382、press then release11、lift7、release6、hold2）。
- interactionのobservable条件は別で、左右**同時**接触mask `b=L & R` に5samples以上の連続true runがあること（542–546行）。連続5samplesを名目50msと記載するならサンプル数が正本で、first-to-last spanは40ms。
- 支持面のcontact/non-contactや指の離脱イベントは `_first_sustained_state`（220–240行）で5連続samplesにより確認し、**確認runの先頭sampleへevent時刻を戻す**。確認が成立する第5sampleをevent onsetとする定義ではない。
- `_confirmed_mask`（344–357行）は長さ5以上のtrue run全体を有効化する。
- raw支持contact `S = obj_plane OR obj_target_tile`（279–294行）。`contact_logging_version>=2` は記録済target contactを使用。古い版にはbbox XY overlapかつ物体bbox bottomとtile surface center zの差<=.008mをORするfallbackがある（243–276行）。今回の層化標本74episodesではversion2=65、version3=9でlegacy fallbackは無かったが、全episodeのversion分布は未集計。

#### Weight

- `_weight_descriptor`710–716行: raw `obj_mass` の最初の値により **light: mass<0.2kg、heavy: mass>=0.2kg**。
- `_weight_is_observable`371–383行: window内に5samples以上連続する**支持面非接触**runが存在し、そのrun内の少なくとも1sampleで左右同時接触していること。左右同時接触が同じ5samples続くことまで要求していない。
- この条件がない場合はweight属性を空欄にし、文章は`an object`（heavy/lightなし）にする。他属性で有効なwindowは残す。
- これは接触maskに基づく幾何学的observability proxyであり、「支持面が全重量を負担したこと」を反力で判定しているわけではない。完全支持以外にも、左右把持不足などでweightを省略する。本文の“omitted when the environment fully supports the object”は十分条件としてはよいが完全な定義ではない。

#### Slip / stable

`_interaction_measurements`514–596行、ラベル適用1136–1153行。

各time sample kの物体COMをc_k、left/right finger link位置をl_k,r_kとする。

1. `m_k=(l_k+r_k)/2`、relative position `u_k=c_k-m_k`。
2. 左右同時接触samplesについて、`r_k-l_k` の成分毎medianを取り、unit vectorにして**window固定grasp axis n**とする。
3. 当該windowの最初の同時接触sample aから最後の同時接触sample bまで（両端含む）を評価区間とする。途中の短い接触断は区間から落とさない。接触前approachと最終接触喪失後の自由運動は除く。
4. `q_k=(I-n n^T)u_k`。
5. `d_max=max_(a<=k<=b) ||q_k-q_a||`。**累積移動距離でもend-to-end差でも瞬時速度でもなく、最初の把持相対位置からの最大接線変位**。
6. `v_proxy=d_max / [(step_last-step_first)*.01]`。分母は接触区間ではなくwindow全体のelapsed time、通常.79s。
7. 同時接触run>=5samplesが成立し、`v_proxy>=.005m/s`ならslip、閾値未満はstable。
8. 非observableならinteraction欠損。pure release/release midairでは計測が可能でもinteractionを付けない。accidental dropはeventラベルが優先。

この量はwindow単位の運動proxyであり、contact pointの局所滑り速度を直接計測するものではない。COM相対運動のため物体回転/姿勢変動に影響され得る。nをsampleごとに更新する回転補償も行わない。appendixの数式はproxyとして表現すべき。

細部の実装注意: `_estimate_interaction`599–608行は非有限速度を欠損にするが、実際のmain1147–1153行ではobservable=trueかつ非有限速度だとstableにfallbackする。通常の有効windowでは退化axis等を想定しないが完全再実装ではmainが正本。今回標本では非有限observable例なし。

#### Accidental drop

- generation `make_step.py:175–206`: 把持後monitorが有効（run_genesis_sim.py:1356）かつintentional release前に、左右両方の接触喪失が5samples連続するとearly termination。接触復帰でstreakリセット。
- generation保存時はearly-drop episodeを破棄せず `accidental_drop_*` として保存し1回retry（run_genesis_sim.py:1831–1846）。
- annotationはepisodeのearly-drop flagと、確認済接触喪失eventを含むwindowにinteraction=accidental dropを付ける（1027–1039,1123–1138）。**accidental-drop episodeの全windowがdropラベルになるわけではない**。
- drop時のactionはその時刻の意図したcommandを維持する（665–671）。失敗後の床衝突は通常placeにしない（1029–1033）。
- action評価ではaccidental-drop episodeを全体ごと除外するが、interaction評価には残す（対象metadataにも明記）。

#### Action

- grasp: 最初の左右同時接触を含むwindow（420–439）。ここだけは5samples確認が必須の実装ではなく、同時接触の立ち上がり。
- lift: lift command後の5連続支持面非接触runの先頭を含む（308–316,1024）。
- place: place command後の5連続支持面接触runの先頭を含む。window内後半でpressに入ってもplace優先（685–689）。
- press: 支持面接触あり、少なくとも片指接触あり、下降 `diff(eef_z)<−1e−5m/sample` **または**明示press command有効区間、の共通maskを5samples確認。place onsetより後で、release開始前（386–417）。単に“接触以後”ではない。
- hold: grasp/lift/place/release eventを含まないpre-placement windowのfallback。支持面に5samples以上接触し続けるpassive waiting holdは除外、accidental-drop eventの例外あり（121–137）。空中搬送もhold。
- release: explicit release command開始（その後5samples接触喪失が確認できた場合）をeventとする。commandがないlegacy non-drop episodeではcontact loss eventを使う（1041–1053）。release開始時点のconfirmed支持contactの有無でrelease/release midairを分ける。
- connected labels: grasp then lift、place then release、press then release。学習に残す。place then pressは意図的に作らない。action判定の優先順位は `_determine_action`640–707行が正本。pure releaseとrelease midairはRQ1の6分類時にreleaseへ統合。

#### Support Surface

- place/pressを含むactionでのみ付与し、accidental dropでは付けない（783–823）。
- 確認済target contactに基づき、slope10/20/30° → gentle/moderate/steep、step .005/.01/.02m → small/medium/large。plane contactならfixed flat。
- wobbleはシナリオ名だけで決定せず、windowの `target_surface_angle_deg` の `max-min>.05°` ならwobbling surface。<=.05°ならmedian(|angle|)に最も近い0/10/20/30°へ離散化しflat/gentle/moderate/steep（742–771）。tieは列挙順の小さい角度。
- 通常planeの`a fixed flat surface`と静止wobbleの`a flat surface`は実際に異なる文章カテゴリ。両者をmetadata再現時に無断統合しない。
- placement outcome (toppled/upright) は原稿版では文章に入れない（metadata placement_outcome_included=false）。

### 実データによる小規模照合

実際のbalanced CSV437,244行を読み、action×interactionごとに最大3行（seed129）を採った76rows/74episodesのrawCSVを読み、現実装で再計算した。

- `slip_tangential_speed_mps` の保存値と再計算の最大差9.71445146547012e−17。
- longest bimanual contact runの不一致0。
- mass threshold＋weight observabilityから再計算したweight/欠損の不一致0。
- contact metadata version2=65episodes、version3=9。
- 全balanced CSVのstart%10 !=0は14,383rowsで全てaccidental drop。

全ラベルを全面再生成して照合したわけではない。保存metadataと対応実装の強い整合性を確認するspot audit。

### 再現付録へ加える生成設定

#### 現実装から復元できる設定（過去全収集の厳密provenanceとは分ける）

|設定|値と実装根拠|
|---|---|
|dt|.01s、`run_genesis_sim.py:529–533`、対象版metadataでも保存確認|
|substeps/gravity|明示overrideなし、現engine default 1 / (0,0,−9.81)m/s²、`genesis/options/solvers.py:45–48`|
|rigid solver|明示overrideなし。現default Newton,max50iterations、line-search50,ls_tolerance1e−2、pyramidal friction、torsional/rolling無効、`solvers.py:549–573`。tolerance=Noneはprecision等によるruntime解決であり、数値を推測しない|
|robot|asset `xml/franka_emika_panda/panda.xml`、Pandaのhand IK、左右位置制御|
|object scale/yaw|bbox x幅またはy幅が(.002,.075)mならscale1で90°/0°yaw、両者不可なら小さい水平寸法に対し `.080/(width+.010)` でscale、`run_genesis_sim.py:249–260`|
|density|`rho=200*uniform(1,10)` kg/m³、すなわち[200,2000]、`run_genesis_sim.py:562–566`。この行はHEADから変更なし。ただし実行時density/乱数seedはmetadataに未保存|
|robot rigid friction|1.0、同538行。coup_friction=3.0は別solver coupling用であり、指–剛体object摩擦3.0と書かない|
|support rigid friction|固定flat/step/slope1.0、wobble1.2、同579,600,607行|
|object/plane摩擦|明示frictionなし。Rigid material defaultはNone→assetまたはsolver処理で決まる（`genesis/engine/materials/rigid.py:34–35,93–96`）。coup_friction=0のplaneを“frictionless ground”と解釈しない|
|arm/finger PD|kp[4500,4500,3500,3500,2000,2000,2000,1000,1000],kv[450,450,350,350,200,200,200,100,100]、`run_genesis_sim.py:1264–1265`|
|finger force range|±10N/DOF、同1266–1269。command名current_force=10でも実際の把持はposition control (`master_movement.py:32–63`)。一定10N contact forceとは主張しない|
|grasp/lift/transfer|grasp200steps、lift前hold100steps、通常lift200steps×.00075m/sampleのtarget更新、simpleならtop hold80steps、terrainなら120stepsXY transfer、他は300steps；`run_genesis_sim.py:1335–1458`。現source設定として記載|
|target XY|通常[.20,.70]m各軸をsampleしspawnから>=.14m、robotから.42–.82m、`run_genesis_sim.py:86–90,266–292`。simpleは元位置へ戻す|
|images|現標準固定1080²、wrist1024²、photo interval80simulation steps、任意final capture、`run_genesis_sim.py:49,541–549`/make_step.py:186–196。F/Tと同じ100Hzで画像保存という意味ではない|

上記はmetadataへeffective runtime configurationとして今後保存すべき。現動作再現には未commit差分込みsource snapshot、dependency lock、mesh/asset identifier、episode RNG seed/density/mass/scale/frictionを同梱する。公開予定コードがあるというだけでは、過去値不明問題は解消しない。

#### 対象版metadataと一致する支持面条件

`data/train_08142026/train_08142026_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_metadata.json` の `simulation` は**uniform_descentのwobble追加campaign**。全旧シナリオで固定速度という意味に広げない。

- slopes10/20/30°。step heights .5/1/2cm。50°/5cm/8cmは選択で除外。
- wobble deckサイズ.24×.21×.016m、pivot z=.074m（現sourceとepisode target_tile metadataで確認可能）。単一hinge、軸yawはuniform0–360°、localX offset±.015m、localY offset符号±・絶対値.060–.070m。
- hinge stiffness5–8N m/rad、damping.22–.35N m s/rad、limit±9–14°、deck mass.45–.55kg。metadata実測rangeは上限下限の内側で一致する。
- contact前後ともcommanded descent .04m/s。post-contact40–100steps（.4–1.0s）追加下降。接触に伴う速度変更false。
- campaignのスループットやGPU-hourはここでは計算していない。本MD第5節の収集ログ監査を参照。


## 4. 記述分布・欠損・分割・言語モデルの全数監査

### 根拠ファイルと再集計方法

- 学習CSV root: `/home/user/Genesis/data/train_08142026/`。以下の V は `terrain_floor_wobble_uniform_descent_contact_clean_slip5mms`。
- 均衡化前: `train_08142026_${V}.csv`。均衡化後: `train_08142026_${V}_balanced.csv`。
- 保存metadata: `train_08142026_${V}_metadata.json`（annotation、balanced_distribution、text_embedding、windowing fields）。
- 12 runsのcheckpoint指定: `/home/user/PatchTST/results/paper_classification_multiseed/paper_classification_3seed_20260909/manifest.json` の runs[].checkpoints.best。
- 実験時コードsnapshot: `/home/user/PatchTST/results/paper_classification_multiseed/paper_classification_3seed_20260909/code/PatchTST_self_supervised/`（以下 S）。
- 再集計script: [再集計コード](paper_audit/recompute_data_audit.py)、全数JSON: [全数JSON](paper_audit/data_training_stats.json)、全269文章のbefore/after頻度: [全文章頻度CSV](paper_audit/annotation_frequencies.csv)。再集計コードは現在の3リポジトリのローカル配置を前提とする監査用。移設時は冒頭のG/Pとmanifest内のcheckpointパスを解決する。
- `annotation` の完全一致で記述を数える。欠損は空/NaN。episodeは csv_path、objectはその /csv/ 直下のdirectory。
- 全12 checkpointで normalization.train_samples + validation_samples がCSV全437,244行に一致することをassertした。dataset readerはCSV行順に採用（`S/patchtst_bert.py:80–144`）。今回の版ではruntime filterによる追加dropがなく、split再構築にCSV行を使用できる。

| 資料 | SHA256 |
| --- | --- |
| /home/user/Genesis/data/train_08142026/train_08142026_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_balanced.csv | b4f38c4b43d68848bb084fef6d8e7a0e58719b491039df0f4ec9d8df704c1570 |
| /home/user/Genesis/data/train_08142026/train_08142026_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms.csv | 670caa98df02368e9694a18c824384783aa21af0cf49627abb2728b3c88469e3 |

### 記述種類数・頻度と意味的多様性

| 指標 | 品質除外後・均衡化前 | 最終学習集合 |
| --- | --- | --- |
| windows | 1,531,185 | 437,244 |
| episodes | 40,878 | 40,851 |
| objects | 893 | 892 |
| 完全一致の文章種類 | 269 | 250 |
| 属性tuple種類（action/weight/interaction/support） | 269 | 250 |
| 1文章あたり最少 / 中央値 / 最大 | 1 / 115.0 / 298,017 | 1 / 116.5 / 74,487 |
| 10窓未満の記述種類 | 56 | 53 |
| 100窓未満の記述種類 | 130 | 120 |
| 上位10文章の窓 | 1,262,076 (82.42%) | 261,944 (59.91%) |

最終CSVの250属性tupleに250文章が対応する。これは定型文の属性組合せによる多様性であり、250個の自由文を独立に収集した／paraphraseを評価したという意味ではない。均衡化によって19文章種類と27 episodes、1物体が最終集合から落ちた。全てのrare descriptionを保持したとは書かず、下記transition/accidental-drop窓の保持を記述する。

#### 上位20文章

| 文章（完全一致） | 前 | 後 | 後の全窓比率 |
| --- | --- | --- | --- |
| Grasping an object with stable contact. | 298,017 | 74,487 | 17.04% |
| Holding a light object, letting it slip. | 59,264 | 40,005 | 9.15% |
| Holding a heavy object steadily. | 159,475 | 38,834 | 8.88% |
| Lifting a light object with stable contact. | 46,275 | 31,936 | 7.30% |
| Lifting a heavy object, letting it slip. | 95,383 | 26,922 | 6.16% |
| Pressing an object against a fixed flat surface with secure contact and then releasing it. | 12,007 | 12,007 | 2.75% |
| Accidentally dropped a heavy object while lifting it. | 11,062 | 11,062 | 2.53% |
| Placing a heavy object on a fixed flat surface, letting it slip. | 9,611 | 9,611 | 2.20% |
| Pressing an object against a fixed flat surface and then releasing it, while it slips. | 9,200 | 9,200 | 2.10% |
| Accidentally dropped an object while lifting it. | 7,880 | 7,880 | 1.80% |
| Grasping an object, letting it slip. | 7,649 | 7,649 | 1.75% |
| Releasing an object from the supporting surface. | 8,029 | 7,271 | 1.66% |
| Placing a light object on a fixed flat surface, letting it slip. | 6,656 | 6,656 | 1.52% |
| Releasing a heavy object in midair. | 5,476 | 5,334 | 1.22% |
| Lifting a light object, letting it slip. | 24,001 | 4,701 | 1.08% |
| Placing a light object on a fixed flat surface with secure contact. | 4,677 | 4,677 | 1.07% |
| Pressing an object against a large step, letting it slip. | 4,527 | 4,527 | 1.04% |
| Pressing an object against a small step and then releasing it, while it slips. | 4,404 | 4,404 | 1.01% |
| Pressing an object against a wobbling surface and then releasing it, while it slips. | 4,299 | 4,299 | 0.98% |
| Pressing an object against a medium step and then releasing it, while it slips. | 4,134 | 4,134 | 0.95% |

#### 属性別の全件分布

##### action

| 値 | 前（全窓比率） | 後（全窓比率） |
| --- | --- | --- |
| hold | 724,715 (47.33%) | 84,629 (19.36%) |
| lift | 263,332 (17.20%) | 84,145 (19.24%) |
| grasp | 325,768 (21.28%) | 84,145 (19.24%) |
| place | 75,415 (4.93%) | 75,415 (17.25%) |
| press then release | 60,832 (3.97%) | 60,832 (13.91%) |
| press | 49,553 (3.24%) | 23,797 (5.44%) |
| place then release | 9,214 (0.60%) | 9,214 (2.11%) |
| release | 8,050 (0.53%) | 7,292 (1.67%) |
| release midair | 13,822 (0.90%) | 7,291 (1.67%) |
| grasp then lift | 484 (0.03%) | 484 (0.11%) |

##### weight

| 値 | 前（全窓比率） | 後（全窓比率） |
| --- | --- | --- |
| (missing) | 472,413 (30.85%) | 187,252 (42.83%) |
| heavy | 620,209 (40.51%) | 128,335 (29.35%) |
| light | 438,563 (28.64%) | 121,657 (27.82%) |

##### interaction

| 値 | 前（全窓比率） | 後（全窓比率） |
| --- | --- | --- |
| stable | 902,872 (58.97%) | 211,916 (48.47%) |
| slip | 557,797 (36.43%) | 180,391 (41.26%) |
| accidental drop | 28,985 (1.89%) | 28,985 (6.63%) |
| (missing) | 41,531 (2.71%) | 15,952 (3.65%) |

##### support_surface

| 値 | 前（全窓比率） | 後（全窓比率） |
| --- | --- | --- |
| (missing) | 1,339,952 (87.51%) | 271,767 (62.15%) |
| a fixed flat surface | 72,103 (4.71%) | 54,888 (12.55%) |
| a large step | 20,567 (1.34%) | 20,560 (4.70%) |
| a medium step | 19,364 (1.26%) | 19,352 (4.43%) |
| a small step | 18,291 (1.19%) | 18,287 (4.18%) |
| a wobbling surface | 15,320 (1.00%) | 15,305 (3.50%) |
| a steep slope | 18,148 (1.19%) | 14,015 (3.21%) |
| a moderate slope | 15,375 (1.00%) | 12,532 (2.87%) |
| a gentle slope | 12,028 (0.79%) | 10,501 (2.40%) |
| a flat surface | 37 (0.00%) | 37 (0.01%) |

支持面は欠損以外で9個の実際の記述値がある。a flat surface は静止判定になった可動支持面の値であり、a fixed flat surface と集計上別。評価prototypeや本文で一括してflatと呼ぶ場合は統合の有無を区別する。

### 欠損属性とtransition

欠損は「その属性の教師情報を使用しない」という意味で、負の状態ラベルではない。weightの欠損は187,252/437,244=42.83%、interactionは15,952=3.65%、supportは271,767=62.15%。元CSVのaction列に欠損はない。教師あり6-class化後はtransition 70,530窓のactionだけをmaskするため、`data_audit.json` に action missing=70,530 と出る。この数字は注釈失敗ではない。根拠: `/home/user/PatchTST/PatchTST_self_supervised/classification_taxonomy.py:22–42`、`S/patchtst_supervised.py:135–155,258–276`。

| 欠損するoptional属性（ほかは有効） | 前 | 後 |
| --- | --- | --- |
| support_surface | 969,044 (63.29%) | 166,648 (38.11%) |
| weight, support_surface | 330,898 (21.61%) | 90,535 (20.71%) |
| weight | 114,053 (7.45%) | 88,450 (20.23%) |
| (none) | 75,659 (4.94%) | 75,659 (17.30%) |
| interaction, support_surface | 13,696 (0.89%) | 7,312 (1.67%) |
| weight, interaction, support_surface | 26,314 (1.72%) | 7,272 (1.66%) |
| weight, interaction | 1,148 (0.07%) | 995 (0.23%) |
| interaction | 373 (0.02%) | 373 (0.09%) |

#### action別の欠損数（最終集合）

| action | 窓数 | weight欠損 | interaction欠損 | support欠損 |
| --- | --- | --- | --- | --- |
| grasp | 84,145 | 82,136 | 0 | 84,145 |
| grasp then lift | 484 | 48 | 1 | 484 |
| hold | 84,629 | 287 | 0 | 84,629 |
| lift | 84,145 | 7,880 | 0 | 84,145 |
| place | 75,415 | 5,033 | 844 | 3,313 |
| place then release | 9,214 | 1,991 | 4 | 0 |
| press | 23,797 | 21,948 | 0 | 468 |
| press then release | 60,832 | 60,658 | 520 | 0 |
| release | 7,292 | 7,271 | 7,292 | 7,292 |
| release midair | 7,291 | 0 | 7,291 | 7,291 |

Transitionは 70,530窓（前4.61%→後16.13%）、107文章。grasp then lift=484、place then release=9,214、press then release=60,832で、全て均衡化前後同数。Accidental dropも全28,985窓が保持され、transitionと合わせたmandatory poolは99,515窓。

samplingコード `/home/user/Genesis/src/processing/rigid/create_train_balanced_set.py:377–395` はconnected actionまたはaccidental dropを必須保持。2-action窓を各componentに1回ずつ数え、release midairをreleaseにまとめる effective count は6 action全て84,629（総component count507,774）。これは437,244窓を排他的な6クラスに等分した分布ではない。label欠損最少化後にweight/interactionの不均衡を最少化するMILP（同:295–335）、層内samplingはreplace=False, random_state=42、最終順序もseed42でshuffle（同:483–494）。保存CSVを配布すると再現が明確になる。

通常stride=10 samplesだが、最終集合ではstartが10の倍数ではない窓が14,383あり、全てinteraction=accidental drop。注釈コードにepisode末端へ整列する例外があるため、通常strideと末端窓を分けて書く必要がある。

### GSO train/validation split と時間的重複

対照学習は `S/patchtst_bert_normalized.py:127–139` のtorch.randperm(total_len, torch.Generator().manual_seed(seed))、末尾round(0.1*N)をvalidation。モデル3種は同一seed内で同じsplit。seed42/43/44でsplit、初期化、学習順が変わる。教師ありは `S/patchtst_supervised.py:538–550` のsklearn train_test_split(test_size=0.1, random_state=seed, stratify=action targets)。6 atomic action + ignore=-100の層化で、transitionのactionだけをignoreする。

**object-disjointでもepisode-disjointでもない。** 元CSVに同一(csv_path,start)の重複が0でも、窓どうしがraw samplesを共有する。以下の重複は同じcsv_pathかつ|start_train-start_val|<80と定義し、validation窓ごとに1件以上のtrain窓と共有するかを全数計数した。対照splitは保存seed/コード式から再構築、教師ありは保存split_indices.npzが該当3runにないため保存label_mapsと同じsklearn式から再構築した監査値。元split manifestを実読した値とは区別する。

| split | train窓 / val窓 | val窓でtrainと時間重複 | train / val / 共通episodes | train / val / 共通objects |
| --- | --- | --- | --- | --- |
| contrastive_seed42 | 393,520 / 43,724 | 39,646 (90.67%) | 40,671 / 21,697 / 21,517 | 892 / 868 / 868 |
| contrastive_seed43 | 393,520 / 43,724 | 39,702 (90.80%) | 40,676 / 21,600 / 21,425 | 892 / 869 / 869 |
| contrastive_seed44 | 393,520 / 43,724 | 39,672 (90.73%) | 40,679 / 21,550 / 21,378 | 892 / 873 / 873 |
| supervised_seed42 | 393,519 / 43,725 | 39,723 (90.85%) | 40,705 / 21,536 / 21,390 | 892 / 873 / 873 |
| supervised_seed43 | 393,519 / 43,725 | 39,612 (90.59%) | 40,681 / 21,580 / 21,410 | 891 / 867 / 866 |
| supervised_seed44 | 393,519 / 43,725 | 39,651 (90.68%) | 40,676 / 21,565 / 21,390 | 891 / 871 / 870 |

seed42対照のvalidation 43,724窓のうち39,646窓（90.67%）にtrainとのraw sample重複がある。29,658窓は最も近いtrain開始点との距離が10以下（70/80サンプル以上を共有）である。このvalidationは学習集合内のモデル選択用であり、未使用物体への汎化検証とは書かない。YCBをcheckpoint選択に使っていないという独立性と、GSO validationにこの重複があることは両方明示する。

入力正規化はsplit後のtrain windowsのみから軸ごとにmeanとpopulation stdを計算し、scale=max(std,1e-6)。ただし重複した時刻はwindow内に登場する回数だけ計数される。対照seed42–44ではcount=393,520×80=31,481,600、教師ありでは31,481,520。raw時刻をdeduplicateした統計ではない。根拠: `/home/user/PatchTST/PatchTST_self_supervised/force_normalization.py:10–35`、checkpoint.normalization.count。

### Sentence-BERT と対照損失の再現条件

- 保存metadataの確定値: `sentence-transformers/all-mpnet-base-v2`、出力768次元、L2 normalizeあり。
- 現在のembedding生成コード `/home/user/Genesis/src/processing/rigid/3_bert_train.py:13–40` はmax_seq_length=512、batch64、sorted unique annotationを一度ずつencodeしてCPU tensorとして保存。default model設定の384とは異なり上書きしている。
- 実物250 tensorsを全読し、shape=(250,768)、全値finite、L2 norm範囲0.9999998808–1.0000001192を確認。
- 語彙ID対応は `annotation_index_map_${V}.csv` のemb_index、実物tensorは `bert_emb_${V}/<emb_index>.pt`。学習中に文encoderは更新されず、事前計算tensorを読み込む。
- ローカルHF cacheにはrevision `e8c3b32edf5434bc2275fc9bab85f82640a19130` があり、modulesはTransformer→mean-token Pooling→Normalize。cache保存物から過去embeddingとの再encode/hash比較は未実施なので、このrevisionを当時使用確定とは断言しない。厳密な再現の正本として保存embeddings/indicesを配布する。
- 現在のGenesis venv package versionsは sentence-transformers5.1.0、transformers4.56.1、torch2.9.1、tokenizers0.22.0。ただし実行当時のfreezeとは未照合のため当時環境の確定値としてappendixへ転記しない。

保存9対照checkpoint（3 architecture×3 seed）全て loss_type=infonce、learnable_temp=False。`S/patchtst_bert.py:186–242` により固定tau=0.07、normalized cosine logit、force→textとtext→forceのcross entropyの平均。対角pairのみpositiveで同一文章のoff-diagonal pairをmaskせず分母に含める。現在コードにmulti-positive variantsがあってもこの実験では未使用。

250文章が437,244窓に再利用され、ランダムに異なる2窓を引いたとき同一文章である確率は sum_c n_c(n_c-1)/(N(N-1))=0.05984745。この分布から512窓batchのanchorあたり同一文章の別窓数の期待値は30.58。これは理論的期待値であり実際の各epoch batchのログ集計ではない。少なくともB=512>250なのでfull batchには必ず文章の重複がある。同じ文をnegativeとするlossの構造を隠さず記載し、multi-positive変更の効果を主張するには別実験が必要。

保存共通学習設定: batch512、max400 epochs、warmup10、peakLR0.002、minLR1e-5、weight_decay0、gradient_clipping1、early_stopping_patience75、early_stopping_min_delta1e-4。optimizerはAdam（`S/patchtst_bert_normalized.py:200`付近）でありAdamWと書かない。全runのbest epoch/安全なallowlist設定はJSON run_checks参照。


## 5. 計算資源・時間・除外率

### 計算資源と時間の書き方

著者の回答「A6000×2、断続的に追加したため所要時間は出せないかも」を記録し、保存ログと照合した。対象ログ内のGenesis表示は `NVIDIA RTX A6000`, `gs.cuda`, device memory `47.41 GB`。これはログの表示値で、GPUを現在照会して過去構成へ代入したものではない。

初期batchの同じログ中に `parallel workers: 2`（4行目など）と `parallel workers: 10`（842038行目以降）がある。最終launcherの10 workersを、歴代全収集に遡って適用しない。一部の追加収集はGPU0だけを使用。CPU型番・RAM・当時driver/CUDA版・各campaignの実効worker上限は一式として保存確認できていない。

| campaignのログ（`data/`以下） | ログ上の開始→終了（JST） | 外側timestampの経過区間 | GPU ID | 解釈 |
| --- | --- | --- | --- | --- |
| rigid_two_rounds_08142026_20260814.log | 8/14 17:07:04 → 8/16 14:27:38 | 45h20m34s | 0,1 | 中断・複数再起動・2→10 workers変更を含む。実稼働時間とはしない。 |
| train_08142026_firm_gentle_two_more_gpu0.log | 8/17 01:09:24 → 10:30:59 | 9h21m35s | 0 | trainのfirm/gentle追加。全dataset生成時間ではない。 |
| rigid_incline20_smallsteps_08142026.log | 8/18 20:44:40 → 8/19 05:53:40 | 9h09m00s | 0,1 | train/evalの追加campaign。 |
| rigid_steps_floor_descent_08142026.log | 8/20 12:26:24 → 22:13:40 | 9h47m16s | 0,1 | train/evalのfloor-descent追加campaign。 |
| rigid_wobble_uniform_descent_capture_08142026.log | 8/23 23:34:55 → 8/24 04:03:42 | 4h28m47s | 0,1 | 最終版で使用したwobble追加。起動・restart込みの限定されたwall-clock例としてappendixに記載。 |

元の `rigid_wobble_capture_08142026.log` は旧wobble版であり、最終datasetでは除外される。この費用を最終採用データの生成費用へ機械的に足さない。上の区間を足して「全生成時間」とも「GPU-hours」ともしない。収集の合間、重複作業、再試行、mesh処理、画像出力、版更新で廃棄されたデータが混在するからである。

ログから抽出したイベント行番号・GPU表示・worker記録は [generation_log_audit.json](paper_audit/generation_log_audit.json)。uniform-descentの開始は1行目、restartは8811行目、終了は829232行目。4h28m47sはこれらの壁時計差だけを意味し、GPU利用率・accepted windows/sを計測した値ではない。

### 母数を明示した主要除外率

GSO/YCBそれぞれの版別metadataの`annotation`を使用する。このcandidate母数はシナリオ選択後のannotation対象であり、全物体の全simulation起動回数でも、生データフォルダ総数でもない。

| 段階 | GSO | YCB |
| --- | ---: | ---: |
| 候補episode | 56,117（100%） | 4,280（100%） |
| rollout品質異常で除外 | 12,567（22.39%） | 1,390（32.48%） |
| annotation可能な窓なし | 2,672（4.76%） | 717（16.75%） |
| annotation非空episode | 40,878（72.84%） | 2,173（50.77%） |
| 非空annotation窓 | 1,531,185 | 111,355 |
| 分布調整後の学習窓 | 437,244（1,531,185の28.56%） | 該当なし |
| 分布調整後の学習episode | 40,851（40,878から27減） | 該当なし |

`empty_annotation_episodes=15,239/2,107`は、rollout異常と窓なしの和。これを別の追加除外項目として足さない。原因ごとの品質異常は複数条件に該当し得るため、metadataにない内訳を推測しない。

rollout異常は、各指Fx/Fy/Fzのいずれかの絶対値>=100 N、COM非有限、COM 1step変位>0.05 m、COM水平半径>2 m、COM zが[-0.1,2.0] m外。100 Nは力ベクトルのnormでもトルク閾値でもない。50°斜面・5/8 cm段差・uniform_descentでない旧wobbleの選択除外は候補母数より前なので、上の22.39%へ混ぜない。

uniform wobbleのtrain側metadataには3,090 terminal invocations、retry込み5,245 saved episodes、accidental-drop episodes 3,946、successful wobble episodes 1,299が記録される。ただしdropは学習に残す物理事象であり、3,946/5,245を「破棄率」としない。logでは全体passがstatus=0でも内部failed workerの警告があるので、launcher正常終了率を全episode成功率と解釈しない。

### 追加記録すると再現性の確認に有用な情報

全収集時間を後付けで作る代わりに、今後の代表batchについて、開始/終了timestamp、GPU数・型番、実worker数、画像保存on/off、cold/warm mesh cache、attempt/retry/accepted counts、simulation/annotation/embeddingそれぞれのwall-clock、accepted episodes/hを保存する。代表batchの測定は既存dataset全体の実績と区別する。今回、新たな生成benchmarkは実行していない。

## 6. RQ1–RQ3の実験結果・前処理の照合

ここでの原稿行番号は改稿前の参照番号。確定した記述修正は第1節に記載済みであり、下記数値の確認は保存集計との照合である。

### 掲載数値の照合結果

#### RQ1

`/home/user/PatchTST/results/paper_classification_multiseed/paper_classification_3seed_20260909/summary.md` の best 行は原稿表の全値と一致。
3 seeds は42/43/44、標本SD (ddof=1)。split・初期値・data orderすべてがseedに応じて変わる。contrastiveはvalidation loss最小、supervisedはvalidation平均Macro F1最大。

SSL表も paths の集計と一致し、1 pretrained seed42＋1 linear-probe seedの結果。SSLと対照学習でreadoutの監督条件が違うことは現在のL277で適切に注意している。

MLP action最高、CNN weight最高、MLP interaction最高は平均値に関して正しい。有意差を示す主張はない。

#### RQ2固定面

`/home/user/ur3_robotiq_ros2/artifacts/fixed_support_model_vs_fz_baseline/summary.csv` の掲載delay:

- rectangle Token-Fusion: 347.8927256 ms、20/20。
- rectangle Fz: 491.99597655 ms、20/20。
- cylinder Token-Fusion: 430.0307830526316 ms、contact付き19/19、全attemptsでは19/20。
- cylinder Fz: 546.84945925 ms、20/20。
- baseball Token-Fusion: 630.9948275454545 ms、11/20。
- baseball Fz: 0/20。

よって本文・abstractの固定面50/60対40/60は、コルクの未記録接触失敗を失敗として数える規則なら整合。

元コルク失敗の決定的根拠:
`/home/user/ur3_robotiq_ros2/data/place_cylinder/20260827_162707_684887/episode_000005/model_evaluation.json:3` は status=failure、L4 outcome=model_place_detected_before_enter、L5 floor_contact_recorded=false、L13 contact_to_halt_ms=null、L15 model_triggered=true。

固定rectangle前半11試行はtimeout0.5秒、後半9は0.8秒。全前半成功が0.479秒以内で原稿成否への影響なし。根拠は `artifacts/fixed_support_model_vs_fz_baseline/manifest.json:7` の exception、および前半各model_evaluation_config。

#### RQ2wobble

`data/place_rectangle_wobble/` の主3 runsに20評価JSON、12成功、平均436.867 ms（paths の精度）を確認。
`data/place_rectangle_wobble_fz_baseline/20260827_234339_773604` は11評価JSON、全timeout0.8秒失敗。
`data/place_rectangle_wobble_fz_baseline_long_horizon/20260829_142525_572228` は20評価JSON、15成功、5timeout、成功mean3889.7273094 ms、最小3342.18981 ms。

したがって本文long-horizon15/20・3.890秒は一致。short-horizon0/20の出所を同定したわけではない。

#### RQ3

正本:
`/home/user/PatchTST/results/terminal_insertion_supervised/terminal_insertion_supervised_20260913_172852/pretrained_linear_vs_scratch_end_to_end_aggregate.csv`
`/home/user/PatchTST/results/terminal_insertion_supervised/terminal_insertion_supervised_20260913_172852/pretrained_lora_vs_end_to_end_aggregate.csv`

原稿L402–405と418の数値は全て一致。
FrozenのN10最高はVAE93.58%、scratch94.51%なので「N10でscratchが最良frozen超え」も正しい。
LoRA N1最高平均masked92.22±0.72%、contrastive91.66±2.77%、supervised91.34±3.15%。

ただし「adaptationは概して全SSLにも有利」とは言えない例:

| 方式・N | Frozen [%] | LoRA [%] |
| --- | ---: | ---: |
| VAE, N5 | 91.11 | 90.57 |
| VAE, N100 | 97.43 | 96.76 |
| VICReg, N1 | 83.62 | 81.52 |
| VICReg, N100 | 96.22 | 95.28 |
| Masked, N50 | 96.79 | 96.68 |

5 seeds は下流seeds42–46で、事前学習を5回繰り返したものではない。`SUMMARY.md:3` が明示する。

### LoRAとモデル構造の確認

`/home/user/PatchTST/PatchTST_self_supervised/token_fusion_encoder.py:33` は `(80-10)//10+1=8` patches。8×12=96 tokenという本文は正しい。
同ファイルL34–65: backbone 3層16 heads、d_model128、d_ff768、fusion1層8 heads、projection 128→128→768。
モデル入力はraw12軸、deltaなし。

`train_terminal_insertion_probe.py:262` の apply_attention_qv_lora は全encoderをfreezeし、時間backboneの W_Q/W_V とpoolerのpacked QKVのQ/Vだけにparametrizationを入れる。
`.../terminal_insertion_supervised_20260913_172852/lora_r4/seed_42/n_001/metrics.json` と `pretrained_adaptation/supervised/lora/seed_42/n_001/metrics.json` の両方で以下8targetsを確認:

```
backbone.encoder.layers.{0,1,2}.self_attn.W_Q.weight
backbone.encoder.layers.{0,1,2}.self_attn.W_V.weight
pooler.encoder.layers.0.self_attn.in_proj_weight[Q]
pooler.encoder.layers.0.self_attn.in_proj_weight[V]
```

1 mapあたりrank4の128×4＋4×128=1,024、8 maps=8,192。binary linear head768+1=769。計8,961 trainable。全学習scratch1,111,937も保存集計と一致。

### 空荷補正の追加監査（重要）

全採用 `model_evaluation_config.json` と `empty_offset_correction.json` の有無・enabled値を集計:

| 元run | config件数 | offsetキー/JSON | normalization |
| --- | ---: | --- | --- |
| place_rectangle_retest/20260827_152109_994319 | 11 | 全件なし/なし | 旧10試行 |
| place_rectangle_retest/20260827_153945_906240 | 9 | 全件なし/なし | 旧10試行 |
| place_cylinder/20260827_162707_684887 | 20 | 全件なし/なし | 旧10試行 |
| place_baseball/20260827_164827_013733 | 9 | 全件なし/なし | 旧10試行 |
| place_baseball/20260827_170059_207659 | 11 | 全件なし/なし | 旧10試行 |
| place_rectangle_wobble/20260827_230202_819708 | 3 | 全件enabled=true/あり | 旧10試行 |
| place_rectangle_wobble/20260827_230524_074937 | 12 | 全件enabled=true/あり | 旧10試行 |
| place_rectangle_wobble/20260827_231650_521705 | 5 | 全件enabled=true/あり | 旧10試行 |
| place_rectangle_add_wobble/20260827_233058_989610 | 13 | 全件enabled=true/あり | 旧10試行 |
| place_rectangle_cnn_norm_remount_20260910/20260910_115223_636682 | 20 | 全件enabled=true/あり | 新10試行 |
| place_rectangle_mlp_norm_remount_20260910/20260910_121547_246046 | 20 | 全件enabled=true/あり | 新10試行 |

`/home/user/ur3_robotiq_ros2/artifacts/cnn_place_input_audit_20260910/README.md` の「なお8/27午後の元の固定面20件は空荷補正導入前」という既存監査とも整合。

旧normalizationは `artifacts/pick_place_normalization/pick_place_normalization.json`、新は `artifacts/pick_place_normalization_remount_20260910/pick_place_normalization.json`。
補正は単にrawゼロにするtareとは違い、
`normalized = (raw * coordinate_sign + reference_empty - current_empty - real_mean) / real_scale`。
ここでcoordinate_sign=[-1,-1,+1,-1,-1,+1]×2、current_emptyは把持前1秒平均、reference_emptyは校正集合の空荷参照。

Fzルールのtareは別手順で、各試行の把持前1秒の各Fzのmedianを引いた和が0未満。確認例: `data/place_rectangle_wobble_fz_baseline/20260827_234339_773604/episode_000001/fz_baseline_evaluation_config.json:7`–L15。

### ユーザー補足後のrectangle追加探索

木製物体名がrectangleであるとの補足を受け、`ur3_robotiq_ros2` 本体・artifactsとsymlink実体のdataを追加探索した。

- retest20、wobble20、add_wobble13の全model_scores.csvヘッダを調査したが、hold_score/place_scoreのみ。fixed/wobbling表面スコア列はない。
- `place_rectangle_surface_selection` は別の二面試行。26 evaluation JSON、最終 `20260829_164758_513312` の20試行はsuccess4/failure16。ここではfixed_surface_score/wobbling_surface_scoreを保存するが、本文の「成功contact固定20＋wobble20」と同じ評価ではない。
- `place_rectangle_wobble_baseline_video` 追加7件＋孤立episode1件は全て5秒timeout設定。前者内訳は早期検知5、成功1（3.744855秒）、max_lower_duration失敗1。これを短時間11件に足しても原稿の0/20の根拠にならない。
- 31/40、77.5、0.263/0.145/0.061/0.081の組合せを裏付ける集計ファイル・スクリプトは未特定。

このため残る再現性確認事項は、support40件の確定採用リスト、読み出すwindowの時刻と前処理、2 prototypeのembedding/hash、試行別cosine CSV、集計コマンド。現ログから任意の時刻を選び元表に近付けることはしない。


## 7. 再現性のために補完する情報

### 必須の未確定事項

| 優先度 | 項目 | 必要な一次情報／解消方法 |
| --- | --- | --- |
| 必須 | 支持面31/40とcosine平均4値 | 確定40 episode IDs、読み出しwindow終端時刻、使用normalization・offset設定、2 prototypeの文字列とtensor、試行別scores、集計command。成功12 wobble＋追加8の候補だけでは確定しない。 |
| 必須 | 短時間wobble Fz 0/20 | 採用20 episode IDs。専用短時間11件と、別の5秒20件を同一cohortと仮定しない。原票が11件しかないなら、11件の評価としての報告へ変更するか、別cohortでの再評価であることを明記して結果を再作成する必要がある。 |
| 必須 | ビルド素材 | ieeeconf.cls、ref.bib、参照JPG2枚、利用TeX環境。下の検証節を参照。 |
| 記述上重要 | 実機トルク原点・左右軸 | MMS101測定原点、取付図、座標軸図、Panda link原点との対応。現実装にはlever-arm補正がないため、同一原点への変換済みとは書かない。 |
| 主張を復活するなら必要 | 反応時間300 msの“独立測定” | 被験者/回数/計測法/平均・ばらつき・生時刻。現改稿では固定補正値としてのみ記載。 |
| 厳密な履歴再現 | text model revision、依存環境、生成source版 | 当時snapshot/lock/revision。現在のHF cacheやpip versionを過去の確定値に代用しない。保存embedding・CSVを公開することでその成果物を再利用可能にする。 |

GSO validationの重複は、記述上の曖昧さを消すだけでなく、可能ならepisodeまたはobject単位で分割した学習・モデル選択を再実施すると強い検証になる。その場合はsplit、正規化、checkpoint選択、RQ1–RQ3へ転用するencoderが変わるので、元の数値をそのまま「重複を除いた結果」として扱えない。今回は原稿監査・改稿の範囲であり、その再実験はしていない。

### 公開物に同梱するとよいもの

- dataset版名・CSV hash、元episode IDとwindow start、train/validation/test manifest。評価3タスク間のwindow再利用も追える形にする。
- 全文章・embedding ID・頻度と保存tensor。属性欠損は空欄の意味、transitionはsupervised action lossのignore対象であることを明記する。
- 仮想F/Tの式・左右軸・トルク原点、使用robot/object meshの識別子、実機側取付寸法と前処理。
- solverの実効設定、質量・密度・scale・摩擦・RNG seed・support parameter、各episodeのcontact_logging_version、コードsnapshotとdirty diff、依存lock。
- checkpoint選択ルール、元pretrained checkpoint、下流nested episode subset、normalizationとその母集団、prototype文字列・hash。
- attempt/retry/除外/採用の台帳、reasonごとの判定順と分母、保存されているgeneration campaignログ。
- 試行別予測、イベント時刻、判定閾値、timing correction、図表生成スクリプト。symlink先の実体を含め、古いabsolute rootを再配置できるようにする。

**公開前の実データ確認事項：** checkpointの保存引数に認証情報のfieldが含まれていた。checkpoint設定を丸ごと論文・公開manifestへコピーせず、必要な学習設定だけを抽出する。公開checkpointも機密fieldを除いた配布版を作る必要がある。本監査JSONはallowlistで必要な設定だけを抽出しており、認証値は転載していない。

## 8. 今回の検証と再集計資料

全数のCSV監査と6通りのsplit再構成は [data_training_stats.json](paper_audit/data_training_stats.json)、再計算コードは [recompute_data_audit.py](paper_audit/recompute_data_audit.py)、完全な頻度表は [annotation_frequencies.csv](paper_audit/annotation_frequencies.csv)。現ローカル配置で以下を実行するとCSV・split・embeddingの集計を更新する。元データ・checkpointは読み取りのみで、出力先は`paper_audit`内の監査JSON/CSVである。

```bash
cd /home/user/Genesis
.venv/bin/python paper_audit/recompute_data_audit.py
```

保存checkpointに対応した環境のtorch/pandas/numpy/scikit-learnを使用する。保存split manifestからの読取と、seed＋分割式からの再構成は区別されている。入力CSVのhashを先に確かめ、異なる版へ無言で適用しない。

仮想ラベルは76窓/74episodesの層化標本でrawから再計算し、weight/欠損・最長同時接触runは全件一致、slip速度の最大差は9.7×10⁻¹⁷。全250個のtext tensorは768次元・finite・unit normを確認。RQ1/RQ3掲載数値は保存集計と照合し、RQ2の固定面・wobble主要検知は試行JSON/保存集計と照合した。

最終静的検証は [verification.json](paper_audit/verification.json)。LaTeXのlabel重複・未解決内部参照0、environmentとbraceの対応正常、文章頻度269行と前後の総数一致、監査JSONおよび再集計Pythonの構文正常を確認した。改稿差分のwhitespace検査も異常なし。

**PDF組版は未実施。** 現環境ではTeX engineと`ieeeconf.cls`、`ref.bib`が見つからず、原稿参照の`fig/overview.jpg`と`fig/ft_compare.jpg`も欠ける。別図への無断置換や仮bibliographyでの代用は行わない。したがって改稿後のページ数・はみ出し・float配置、および欠損JPGとcaptionの最終一致は未検証。引用文献の書誌・関連研究の全主張の独立照合も本監査の完了範囲には含めない。

`paper.tex`と頻度CSVは既存のGit ignore対象。通常の`git diff`に原稿の変更が現れないため、作業前原稿との直接差分で検査した。配布・version管理へ含める際はこれらの実ファイルも対象にする。

## 9. 全記述の前後頻度

品質除外後269種類。最終250種類、最終頻度0の19種類も表に残す。前の母数1,531,185窓、後437,244窓。完全一致の文章を数え、異なるsupport phraseや句読点を統合しない。

| 文章 | 前 | 後 | 後比率 |
| --- | --- | --- | --- |
| Grasping an object with stable contact. | 298017 | 74487 | 17.0356% |
| Holding a light object, letting it slip. | 59264 | 40005 | 9.1494% |
| Holding a heavy object steadily. | 159475 | 38834 | 8.8815% |
| Lifting a light object with stable contact. | 46275 | 31936 | 7.3039% |
| Lifting a heavy object, letting it slip. | 95383 | 26922 | 6.1572% |
| Pressing an object against a fixed flat surface with secure contact and then releasing it. | 12007 | 12007 | 2.7461% |
| Accidentally dropped a heavy object while lifting it. | 11062 | 11062 | 2.5299% |
| Placing a heavy object on a fixed flat surface, letting it slip. | 9611 | 9611 | 2.1981% |
| Pressing an object against a fixed flat surface and then releasing it, while it slips. | 9200 | 9200 | 2.1041% |
| Accidentally dropped an object while lifting it. | 7880 | 7880 | 1.8022% |
| Grasping an object, letting it slip. | 7649 | 7649 | 1.7494% |
| Releasing an object from the supporting surface. | 8029 | 7271 | 1.6629% |
| Placing a light object on a fixed flat surface, letting it slip. | 6656 | 6656 | 1.5223% |
| Releasing a heavy object in midair. | 5476 | 5334 | 1.2199% |
| Lifting a light object, letting it slip. | 24001 | 4701 | 1.0751% |
| Placing a light object on a fixed flat surface with secure contact. | 4677 | 4677 | 1.0697% |
| Pressing an object against a large step, letting it slip. | 4527 | 4527 | 1.0353% |
| Pressing an object against a small step and then releasing it, while it slips. | 4404 | 4404 | 1.0072% |
| Pressing an object against a wobbling surface and then releasing it, while it slips. | 4299 | 4299 | 0.9832% |
| Pressing an object against a medium step and then releasing it, while it slips. | 4134 | 4134 | 0.9455% |
| Placing a heavy object on a fixed flat surface with secure contact. | 3793 | 3793 | 0.8675% |
| Pressing an object against a large step and then releasing it, while it slips. | 3749 | 3749 | 0.8574% |
| Pressing an object against a medium step, letting it slip. | 3454 | 3454 | 0.7899% |
| Accidentally dropped a heavy object while holding it. | 3337 | 3337 | 0.7632% |
| Placing a heavy object on a large step, letting it slip. | 3114 | 3114 | 0.7122% |
| Pressing an object against a large step with secure contact and then releasing it. | 2968 | 2968 | 0.6788% |
| Pressing an object against a steep slope, letting it slip. | 7048 | 2947 | 0.6740% |
| Placing a heavy object on a medium step, letting it slip. | 2879 | 2879 | 0.6584% |
| Pressing an object against a small step, letting it slip. | 2855 | 2855 | 0.6530% |
| Pressing an object against a medium step with secure contact and then releasing it. | 2610 | 2610 | 0.5969% |
| Placing a heavy object on a small step, letting it slip. | 2590 | 2590 | 0.5923% |
| Pressing an object against a fixed flat surface with secure contact. | 2570 | 2570 | 0.5878% |
| Pressing an object against a moderate slope with secure contact and then releasing it. | 2451 | 2451 | 0.5606% |
| Pressing an object against a steep slope with secure contact and then releasing it. | 2397 | 2397 | 0.5482% |
| Pressing an object against a small step with secure contact and then releasing it. | 2337 | 2337 | 0.5345% |
| Placing a heavy object on a wobbling surface, letting it slip. | 2314 | 2314 | 0.5292% |
| Accidentally dropped a heavy object while placing it. | 2267 | 2267 | 0.5185% |
| Pressing an object against a gentle slope and then releasing it, while it slips. | 2262 | 2262 | 0.5173% |
| Pressing an object against a wobbling surface with secure contact and then releasing it. | 2018 | 2018 | 0.4615% |
| Releasing a light object in midair. | 8170 | 1957 | 0.4476% |
| Pressing an object against a moderate slope and then releasing it, while it slips. | 1927 | 1927 | 0.4407% |
| Pressing an object against a moderate slope, letting it slip. | 4747 | 1921 | 0.4393% |
| Placing a light object on a large step, letting it slip. | 1807 | 1807 | 0.4133% |
| Pressing an object against a steep slope and then releasing it, while it slips. | 1788 | 1788 | 0.4089% |
| Placing a light object on a medium step, letting it slip. | 1776 | 1776 | 0.4062% |
| Grasping a light object with stable contact. | 1684 | 1684 | 0.3851% |
| Accidentally dropped a light object while lifting it. | 1644 | 1644 | 0.3760% |
| Placing a heavy object on a moderate slope, letting it slip. | 1626 | 1626 | 0.3719% |
| Placing a light object on a small step, letting it slip. | 1608 | 1608 | 0.3678% |
| Placing an object on a fixed flat surface, letting it slip. | 1603 | 1603 | 0.3666% |
| Placing a light object on a wobbling surface with secure contact. | 1590 | 1590 | 0.3636% |
| Pressing an object against a gentle slope with secure contact and then releasing it. | 1587 | 1587 | 0.3630% |
| Placing a heavy object on a gentle slope, letting it slip. | 1559 | 1559 | 0.3566% |
| Placing a heavy object on a steep slope, letting it slip. | 1539 | 1539 | 0.3520% |
| Placing a heavy object on a wobbling surface with secure contact. | 1400 | 1400 | 0.3202% |
| Placing a light object on a steep slope with secure contact. | 1398 | 1398 | 0.3197% |
| Placing a light object on a large step with secure contact. | 1378 | 1378 | 0.3152% |
| Placing a light object on a medium step with secure contact. | 1333 | 1333 | 0.3049% |
| Placing a light object on a small step with secure contact. | 1330 | 1330 | 0.3042% |
| Placing a light object on a moderate slope with secure contact. | 1275 | 1275 | 0.2916% |
| Placing a light object on a wobbling surface, letting it slip. | 1249 | 1249 | 0.2857% |
| Placing a heavy object on a large step with secure contact. | 1233 | 1233 | 0.2820% |
| Holding a light object steadily. | 258938 | 1172 | 0.2680% |
| Pressing an object against a steep slope with secure contact. | 1133 | 1133 | 0.2591% |
| Placing a light object on a fixed flat surface with secure contact and then releasing it. | 1124 | 1124 | 0.2571% |
| Pressing an object against a gentle slope, letting it slip. | 2642 | 1122 | 0.2566% |
| Placing a heavy object on a steep slope with secure contact. | 1117 | 1117 | 0.2555% |
| Placing a light object on a moderate slope, letting it slip. | 1104 | 1104 | 0.2525% |
| Placing a heavy object on a medium step with secure contact. | 1091 | 1091 | 0.2495% |
| Placing a light object on a gentle slope, letting it slip. | 1076 | 1076 | 0.2461% |
| Placing a heavy object on a small step with secure contact. | 1015 | 1015 | 0.2321% |
| Accidentally dropped a light object while holding it. | 994 | 994 | 0.2273% |
| Placing a light object on a gentle slope with secure contact. | 989 | 989 | 0.2262% |
| Placing a heavy object on a moderate slope with secure contact. | 978 | 978 | 0.2237% |
| Placing a light object on a steep slope, letting it slip. | 948 | 948 | 0.2168% |
| Accidentally dropped a light object while placing it. | 888 | 888 | 0.2031% |
| Placing a light object on a fixed flat surface and then releasing it while it slips. | 791 | 791 | 0.1809% |
| Placing a heavy object on a gentle slope with secure contact. | 767 | 767 | 0.1754% |
| Placing a light object on a small step with secure contact and then releasing it. | 519 | 519 | 0.1187% |
| Placing a light object on a medium step with secure contact and then releasing it. | 480 | 480 | 0.1098% |
| Placing an object on a large step, letting it slip. | 448 | 448 | 0.1025% |
| Placing a heavy object on a wobbling surface and then releasing it while it slips. | 431 | 431 | 0.0986% |
| Placing an object on a steep slope, letting it slip. | 406 | 406 | 0.0929% |
| Pressing an object against a moderate slope with secure contact. | 404 | 404 | 0.0924% |
| Placing a light object on a small step and then releasing it while it slips. | 396 | 396 | 0.0906% |
| Pressing a heavy object against a fixed flat surface, letting it slip. | 396 | 396 | 0.0906% |
| Placing an object on a fixed flat surface. | 391 | 391 | 0.0894% |
| Placing an object on a medium step, letting it slip. | 386 | 386 | 0.0883% |
| Placing an object on a fixed flat surface and then releasing it while it slips. | 372 | 372 | 0.0851% |
| Placing an object on a wobbling surface and then releasing it while it slips. | 359 | 359 | 0.0821% |
| Placing a light object on a wobbling surface and then releasing it while it slips. | 353 | 353 | 0.0807% |
| Placing a heavy object on a fixed flat surface with secure contact and then releasing it. | 352 | 352 | 0.0805% |
| Pressing an object against a wobbling surface, letting it slip. | 345 | 345 | 0.0789% |
| Placing an object on a small step, letting it slip. | 341 | 341 | 0.0780% |
| Placing a light object on a medium step and then releasing it while it slips. | 328 | 328 | 0.0750% |
| Accidentally dropped a heavy object while pressing it. | 328 | 328 | 0.0750% |
| Placing an object on a moderate slope, letting it slip. | 327 | 327 | 0.0748% |
| Grasping then lifting a heavy object, letting it slip. | 326 | 326 | 0.0746% |
| Placing an object on a wobbling surface, letting it slip. | 325 | 325 | 0.0743% |
| Placing a light object on a large step with secure contact and then releasing it. | 323 | 323 | 0.0739% |
| Placing a light object on a fixed flat surface. | 320 | 320 | 0.0732% |
| Placing a heavy object on a fixed flat surface and then releasing it while it slips. | 304 | 304 | 0.0695% |
| Placing a light object on a gentle slope with secure contact and then releasing it. | 295 | 295 | 0.0675% |
| Accidentally dropped an object while holding it. | 287 | 287 | 0.0656% |
| Placing an object on a gentle slope, letting it slip. | 236 | 236 | 0.0540% |
| Placing an object on a fixed flat surface with secure contact and then releasing it. | 228 | 228 | 0.0521% |
| Grasping a light object, letting it slip. | 222 | 222 | 0.0508% |
| Pressing an object against a fixed flat surface and then releasing it. | 210 | 210 | 0.0480% |
| Placing a light object on a large step and then releasing it while it slips. | 197 | 197 | 0.0451% |
| Pressing a heavy object against a small step, letting it slip. | 180 | 180 | 0.0412% |
| Pressing an object against a gentle slope with secure contact. | 173 | 173 | 0.0396% |
| Pressing a heavy object against a large step, letting it slip. | 173 | 173 | 0.0396% |
| Placing an object on a small step and then releasing it while it slips. | 168 | 168 | 0.0384% |
| Placing a heavy object on a small step with secure contact and then releasing it. | 167 | 167 | 0.0382% |
| Pressing an object against a wobbling surface and then releasing it. | 161 | 161 | 0.0368% |
| Accidentally dropped an object while placing it. | 158 | 158 | 0.0361% |
| Placing an object on a medium step and then releasing it while it slips. | 153 | 153 | 0.0350% |
| Pressing an object against a large step with secure contact. | 148 | 148 | 0.0338% |
| Placing a light object on a moderate slope with secure contact and then releasing it. | 146 | 146 | 0.0334% |
| Pressing an object against a wobbling surface with secure contact. | 139 | 139 | 0.0318% |
| Pressing a heavy object against a medium step, letting it slip. | 135 | 135 | 0.0309% |
| Placing an object on a medium step with secure contact and then releasing it. | 125 | 125 | 0.0286% |
| Placing a heavy object on a medium step with secure contact and then releasing it. | 119 | 119 | 0.0272% |
| Placing a heavy object on a small step and then releasing it while it slips. | 119 | 119 | 0.0272% |
| Placing an object on a fixed flat surface with secure contact. | 118 | 118 | 0.0270% |
| Placing an object on a large step and then releasing it while it slips. | 115 | 115 | 0.0263% |
| Accidentally dropped a light object while pressing it. | 113 | 113 | 0.0258% |
| Placing an object on a small step with secure contact and then releasing it. | 111 | 111 | 0.0254% |
| Placing a light object on a gentle slope and then releasing it while it slips. | 108 | 108 | 0.0247% |
| Placing a heavy object on a wobbling surface with secure contact and then releasing it. | 108 | 108 | 0.0247% |
| Placing a heavy object on a medium step and then releasing it while it slips. | 98 | 98 | 0.0224% |
| Pressing an object against a medium step with secure contact. | 96 | 96 | 0.0220% |
| Placing a heavy object on a large step with secure contact and then releasing it. | 91 | 91 | 0.0208% |
| Pressing an object against a small step with secure contact. | 87 | 87 | 0.0199% |
| Placing an object on a steep slope with secure contact. | 85 | 85 | 0.0194% |
| Placing an object on a large step with secure contact and then releasing it. | 83 | 83 | 0.0190% |
| Placing a light object on a wobbling surface with secure contact and then releasing it. | 79 | 79 | 0.0181% |
| Placing an object on a wobbling surface with secure contact and then releasing it. | 77 | 77 | 0.0176% |
| Pressing a light object against a fixed flat surface, letting it slip. | 74 | 74 | 0.0169% |
| Pressing an object against a steep slope and then releasing it. | 72 | 72 | 0.0165% |
| Pressing a heavy object against a gentle slope, letting it slip. | 72 | 72 | 0.0165% |
| Placing a light object on a moderate slope and then releasing it while it slips. | 66 | 66 | 0.0151% |
| Pressing a light object against a medium step, letting it slip. | 59 | 59 | 0.0135% |
| Placing a heavy object on a gentle slope with secure contact and then releasing it. | 59 | 59 | 0.0135% |
| Placing an object on a gentle slope and then releasing it while it slips. | 59 | 59 | 0.0135% |
| Pressing a heavy object against a moderate slope, letting it slip. | 57 | 57 | 0.0130% |
| Pressing a light object against a moderate slope, letting it slip. | 54 | 54 | 0.0124% |
| Grasping a heavy object with stable contact. | 53 | 53 | 0.0121% |
| Placing a heavy object on a large step and then releasing it while it slips. | 51 | 51 | 0.0117% |
| Grasping a heavy object, letting it slip. | 50 | 50 | 0.0114% |
| Grasping then lifting a light object, letting it slip. | 46 | 46 | 0.0105% |
| Placing an object on a moderate slope with secure contact and then releasing it. | 43 | 43 | 0.0098% |
| Placing an object on a moderate slope with secure contact. | 42 | 42 | 0.0096% |
| Placing an object on a gentle slope with secure contact and then releasing it. | 42 | 42 | 0.0096% |
| Grasping then lifting a heavy object with stable contact. | 40 | 40 | 0.0091% |
| Pressing a light object against a steep slope, letting it slip. | 39 | 39 | 0.0089% |
| Placing a light object on a steep slope and then releasing it while it slips. | 34 | 34 | 0.0078% |
| Pressing a heavy object against a fixed flat surface with secure contact. | 32 | 32 | 0.0073% |
| Placing an object on a moderate slope and then releasing it while it slips. | 30 | 30 | 0.0069% |
| Placing an object on a large step with secure contact. | 29 | 29 | 0.0066% |
| Accidentally dropped an object while pressing it. | 27 | 27 | 0.0062% |
| Placing a light object on a flat surface with secure contact. | 26 | 26 | 0.0059% |
| Grasping then lifting an object, letting it slip. | 26 | 26 | 0.0059% |
| Pressing an object against a large step and then releasing it. | 26 | 26 | 0.0059% |
| Grasping then lifting a light object with stable contact. | 24 | 24 | 0.0055% |
| Placing a light object on a steep slope with secure contact and then releasing it. | 24 | 24 | 0.0055% |
| Placing an object on a medium step. | 23 | 23 | 0.0053% |
| Pressing a light object against a large step, letting it slip. | 22 | 22 | 0.0050% |
| Placing a heavy object on a fixed flat surface. | 22 | 22 | 0.0050% |
| Placing a heavy object on a gentle slope and then releasing it while it slips. | 22 | 22 | 0.0050% |
| Pressing a light object against a gentle slope, letting it slip. | 22 | 22 | 0.0050% |
| Grasping then lifting an object with stable contact. | 21 | 21 | 0.0048% |
| Pressing an object against a moderate slope and then releasing it. | 21 | 21 | 0.0048% |
| Placing a heavy object on a moderate slope with secure contact and then releasing it. | 21 | 21 | 0.0048% |
| Placing an object on a large step. | 21 | 21 | 0.0048% |
| Placing an object on a small step. | 20 | 20 | 0.0046% |
| Pressing a light object against a medium step and then releasing it, while it slips. | 20 | 20 | 0.0046% |
| Releasing a light object from the supporting surface. | 18 | 18 | 0.0041% |
| Placing an object on a gentle slope with secure contact. | 18 | 18 | 0.0041% |
| Pressing a light object against a wobbling surface with secure contact and then releasing it. | 18 | 18 | 0.0041% |
| Placing an object on a wobbling surface with secure contact. | 17 | 17 | 0.0039% |
| Placing an object on a steep slope and then releasing it while it slips. | 15 | 15 | 0.0034% |
| Pressing a heavy object against a steep slope, letting it slip. | 15 | 15 | 0.0034% |
| Pressing a light object against a steep slope with secure contact. | 14 | 14 | 0.0032% |
| Pressing a heavy object against a medium step and then releasing it, while it slips. | 14 | 14 | 0.0032% |
| Pressing a light object against a large step and then releasing it, while it slips. | 14 | 14 | 0.0032% |
| Pressing a light object against a fixed flat surface with secure contact and then releasing it. | 13 | 13 | 0.0030% |
| Pressing an object against a medium step and then releasing it. | 13 | 13 | 0.0030% |
| Placing a light object on a large step. | 13 | 13 | 0.0030% |
| Pressing a light object against a large step with secure contact and then releasing it. | 12 | 12 | 0.0027% |
| Placing an object on a steep slope with secure contact and then releasing it. | 10 | 10 | 0.0023% |
| Placing an object on a small step with secure contact. | 10 | 10 | 0.0023% |
| Pressing an object against a small step and then releasing it. | 10 | 10 | 0.0023% |
| Pressing a heavy object against a gentle slope with secure contact. | 10 | 10 | 0.0023% |
| Placing a heavy object on a steep slope with secure contact and then releasing it. | 10 | 10 | 0.0023% |
| Placing an object on a medium step with secure contact. | 10 | 10 | 0.0023% |
| Pressing a light object against a moderate slope with secure contact and then releasing it. | 10 | 10 | 0.0023% |
| Pressing a heavy object against a fixed flat surface and then releasing it, while it slips. | 9 | 9 | 0.0021% |
| Pressing a heavy object against a large step with secure contact. | 8 | 8 | 0.0018% |
| Pressing a light object against a moderate slope with secure contact. | 7 | 7 | 0.0016% |
| Pressing a heavy object against a steep slope with secure contact. | 7 | 7 | 0.0016% |
| Pressing a light object against a small step, letting it slip. | 7 | 7 | 0.0016% |
| Placing a heavy object on a flat surface with secure contact. | 7 | 7 | 0.0016% |
| Pressing a heavy object against a moderate slope with secure contact. | 7 | 7 | 0.0016% |
| Pressing a heavy object against a large step with secure contact and then releasing it. | 7 | 7 | 0.0016% |
| Pressing a heavy object against a fixed flat surface with secure contact and then releasing it. | 7 | 7 | 0.0016% |
| Pressing an object against a gentle slope and then releasing it. | 7 | 7 | 0.0016% |
| Placing an object on a gentle slope. | 6 | 6 | 0.0014% |
| Pressing a light object against a steep slope and then releasing it, while it slips. | 6 | 6 | 0.0014% |
| Pressing a light object against a moderate slope and then releasing it, while it slips. | 6 | 6 | 0.0014% |
| Placing an object on a steep slope. | 5 | 5 | 0.0011% |
| Pressing a heavy object against a wobbling surface and then releasing it, while it slips. | 5 | 5 | 0.0011% |
| Placing an object on a moderate slope. | 4 | 4 | 0.0009% |
| Pressing a light object against a wobbling surface and then releasing it, while it slips. | 4 | 4 | 0.0009% |
| Pressing a heavy object against a small step with secure contact. | 4 | 4 | 0.0009% |
| Pressing a light object against a fixed flat surface and then releasing it, while it slips. | 4 | 4 | 0.0009% |
| Pressing a heavy object against a medium step with secure contact. | 4 | 4 | 0.0009% |
| Placing a light object on a steep slope. | 4 | 4 | 0.0009% |
| Placing an object on a wobbling surface. | 4 | 4 | 0.0009% |
| Pressing a light object against a small step and then releasing it, while it slips. | 4 | 4 | 0.0009% |
| Placing a light object on a medium step. | 4 | 4 | 0.0009% |
| Pressing a heavy object against a gentle slope and then releasing it, while it slips. | 4 | 4 | 0.0009% |
| Pressing a heavy object against a small step and then releasing it, while it slips. | 3 | 3 | 0.0007% |
| Releasing a heavy object from the supporting surface. | 3 | 3 | 0.0007% |
| Pressing a heavy object against a wobbling surface with secure contact and then releasing it. | 3 | 3 | 0.0007% |
| Pressing a light object against a medium step with secure contact. | 3 | 3 | 0.0007% |
| Pressing a heavy object against a gentle slope with secure contact and then releasing it. | 3 | 3 | 0.0007% |
| Placing a heavy object on a moderate slope and then releasing it while it slips. | 3 | 3 | 0.0007% |
| Pressing a heavy object against a large step and then releasing it, while it slips. | 2 | 2 | 0.0005% |
| Pressing a light object against a wobbling surface, letting it slip. | 2 | 2 | 0.0005% |
| Placing a heavy object on a flat surface, letting it slip. | 2 | 2 | 0.0005% |
| Placing a light object on a gentle slope. | 2 | 2 | 0.0005% |
| Pressing a heavy object against a medium step with secure contact and then releasing it. | 2 | 2 | 0.0005% |
| Pressing a light object against a medium step with secure contact and then releasing it. | 2 | 2 | 0.0005% |
| Placing a heavy object on a steep slope and then releasing it while it slips. | 2 | 2 | 0.0005% |
| Placing a light object on a flat surface, letting it slip. | 2 | 2 | 0.0005% |
| Pressing a heavy object against a wobbling surface with secure contact. | 2 | 2 | 0.0005% |
| Placing a light object on a moderate slope. | 2 | 2 | 0.0005% |
| Placing a light object on a fixed flat surface and then releasing it. | 2 | 2 | 0.0005% |
| Pressing a light object against a fixed flat surface with secure contact. | 1 | 1 | 0.0002% |
| Grasping then lifting an object. | 1 | 1 | 0.0002% |
| Placing a heavy object on a medium step. | 1 | 1 | 0.0002% |
| Placing a heavy object on a fixed flat surface and then releasing it. | 1 | 1 | 0.0002% |
| Pressing a light object against a small step with secure contact and then releasing it. | 1 | 1 | 0.0002% |
| Pressing a heavy object against a wobbling surface, letting it slip. | 1 | 1 | 0.0002% |
| Pressing a light object against a wobbling surface with secure contact. | 1 | 1 | 0.0002% |
| Pressing a light object against a gentle slope and then releasing it, while it slips. | 1 | 1 | 0.0002% |
| Placing a light object on a small step. | 1 | 1 | 0.0002% |
| Placing a heavy object on a large step. | 1 | 1 | 0.0002% |
| Placing an object on a wobbling surface and then releasing it. | 1 | 1 | 0.0002% |
| Holding a heavy object, letting it slip. | 241523 | 0 | 0.0000% |
| Lifting a heavy object with stable contact. | 61107 | 0 | 0.0000% |
| Grasping an object. | 18093 | 0 | 0.0000% |
| Pressing an object against a fixed flat surface, letting it slip. | 17156 | 0 | 0.0000% |
| Lifting an object with stable contact. | 13434 | 0 | 0.0000% |
| Lifting an object, letting it slip. | 2531 | 0 | 0.0000% |
| Holding an object, letting it slip. | 729 | 0 | 0.0000% |
| Releasing an object in midair. | 176 | 0 | 0.0000% |
| Holding an object steadily. | 139 | 0 | 0.0000% |
| Pressing an object against a fixed flat surface. | 59 | 0 | 0.0000% |
| Pressing an object against a steep slope. | 32 | 0 | 0.0000% |
| Holding a light object. | 29 | 0 | 0.0000% |
| Pressing an object against a moderate slope. | 17 | 0 | 0.0000% |
| Lifting an object. | 15 | 0 | 0.0000% |
| Pressing an object against a wobbling surface. | 15 | 0 | 0.0000% |
| Pressing an object against a medium step. | 12 | 0 | 0.0000% |
| Pressing an object against a gentle slope. | 7 | 0 | 0.0000% |
| Pressing an object against a large step. | 7 | 0 | 0.0000% |
| Pressing an object against a small step. | 4 | 0 | 0.0000% |

