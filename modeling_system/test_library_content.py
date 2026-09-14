import copy,unittest
from types import SimpleNamespace as NS
from .library_content import _material_tree
from .store import canonical,digest

class Tree(NS):
    def keys(self):return []

class LibraryContentTests(unittest.TestCase):
    def tree(self):
        socket=NS(identifier='Base Color',bl_idname='NodeSocketColor',default_value=[.1,.2,.3,1.],hide=False)
        node=NS(name='Principled',bl_idname='ShaderNodeBsdfPrincipled',bl_rna=NS(properties=[]),inputs=[socket],outputs=[])
        return Tree(is_embedded_data=True,nodes=[node],links=[],animation_data=None)
    def test_default_shader_socket_changes_are_content(self):
        tree=self.tree();before=digest(canonical(_material_tree(tree,lambda x:x.name)))
        tree.nodes[0].inputs[0].default_value[0]+=.01
        self.assertNotEqual(before,digest(canonical(_material_tree(tree,lambda x:x.name))))
    def test_unsupported_external_group_and_node_refuse(self):
        tree=self.tree();tree.is_embedded_data=False
        with self.assertRaisesRegex(ValueError,'shared'): _material_tree(tree,lambda x:x.name)
        tree.is_embedded_data=True;tree.nodes[0].bl_idname='ShaderNodeTexImage'
        with self.assertRaisesRegex(ValueError,'qualification'): _material_tree(tree,lambda x:x.name)

if __name__=='__main__':unittest.main()
