"""Migration boundaries: symlinks stay inside the destination; copies preserve bytes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

class WorkspaceToolTest(unittest.TestCase):
    def test_artifact_attachment_never_copies_or_replaces_data(self):
        import json
        linker=module('link_local')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'bundle';root.mkdir()
            source=Path(temporary)/'ssd';source.mkdir();(source/'data').write_text('keep')
            linker.ROOT=root
            config=root/'config.json'
            config.write_text(json.dumps({'links':[{'source':str(source),'destination':'artifacts/input'}]}))
            linker.attach(config,True)
            target=root/'artifacts/input'
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(),source)
            target.unlink();target.mkdir();(target/'own').write_text('keep too')
            with self.assertRaises(RuntimeError):linker.attach(config,True)
            self.assertEqual((source/'data').read_text(),'keep')
            self.assertEqual((target/'own').read_text(),'keep too')

    def test_materialized_links_are_relative_and_relocatable(self):
        workspace=module('workspace')
        with tempfile.TemporaryDirectory() as temporary:
            first=Path(temporary)/'first';(first/'artifacts').mkdir(parents=True)
            workspace.link(first/'workspace/data',first/'artifacts')
            self.assertFalse((first/'workspace/data').readlink().is_absolute())
            second=Path(temporary)/'second';first.rename(second)
            self.assertEqual((second/'workspace/data').resolve(),second/'artifacts')

    def test_link_never_replaces_real_data(self):
        workspace=module('workspace')
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'existing';path.write_text('keep')
            with self.assertRaises(RuntimeError): workspace.link(path,Path(temporary)/'target')
            self.assertEqual(path.read_text(),'keep')

if __name__=='__main__': unittest.main()
