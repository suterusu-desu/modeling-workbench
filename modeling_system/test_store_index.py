"""Fresh retrieval with fewer unrelated receipt opens; never cached evidence."""
import builtins
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .store import Store, canonical


class RecordIndexTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.store=Store(Path(temp.name)/'store')

    def test_warm_scan_reuses_only_headers_and_verifies_selected_bytes(self):
        selected=self.store.put('lesson', {'fact':'verified'})
        for n in range(24):self.store.put('receipt', {'number':n})
        with patch('modeling_system.store.open', wraps=builtins.open) as opened:
            first=list(self.store.records(('lesson',)))
            self.assertEqual(opened.call_count,25)
            opened.reset_mock()
            with patch.object(self.store,'get',wraps=self.store.get) as verified:
                self.assertEqual(list(self.store.records(('lesson',))),first)
                verified.assert_called_once_with(selected)
            self.assertEqual(opened.call_count,0)

    def test_new_records_from_another_process_and_deletions_are_seen(self):
        old=self.store.put('lesson',{'fact':'old'})
        list(self.store.records(('lesson',)))
        other=Store(self.store.root);new=other.put('lesson',{'fact':'new'})
        self.assertEqual({k for k,_ in self.store.records(('lesson',))},{old,new})
        (self.store.root/'records'/(old+'.json')).unlink()
        self.assertEqual([k for k,_ in self.store.records(('lesson',))],[new])

    def test_mutation_cannot_be_hidden_by_a_warm_header_or_restored_timestamp(self):
        key=self.store.put('lesson',{'fact':'original'})
        list(self.store.records(('lesson',)))
        path=self.store.root/'records'/(key+'.json');before=path.stat()
        data=json.loads(path.read_bytes());data['payload']['fact']='modified'
        path.write_bytes(canonical(data));os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'integrity'):
            list(self.store.records(('lesson',)))

    def test_caller_cannot_mutate_a_later_verified_result(self):
        key=self.store.put('lesson',{'fact':['original']})
        rows=list(self.store.records(('lesson',)));rows[0][1]['fact'].append('caller')
        self.assertEqual(list(self.store.records(('lesson',))),[(key,{'fact':['original']})])
        self.assertEqual(list(Store(self.store.root).records(('lesson',))),[(key,{'fact':['original']})])


if __name__=='__main__':unittest.main()
