import unittest
import lupa
import os
import json

class TestSDDjdslParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lua = lupa.LuaRuntime(unpack_returned_tuples=True)
        
        # We need to construct the parser by running the lua file.
        # Since the lua file ends with 'return M', we can execute it to get the Lua table.
        # However, Lua "dofile" might fail if paths are weird, so let's load it directly:
        parser_path = os.path.join(os.path.dirname(__file__), "..", "..", "extension", "scripts", "sddj_dsl_parser.lua")
        parser_path = os.path.abspath(parser_path)
        with open(parser_path, "r", encoding="utf-8") as f:
            lua_code = f.read()
            
        cls.parser = cls.lua.execute(lua_code)
        
    def test_empty_string(self):
        res = self.parser.parse("", 100, 24)
        self.assertEqual(len(res.keyframes), 0)
        self.assertEqual(res.default_prompt, "")

    def test_whitespace_string(self):
        res = self.parser.parse("   \n \t  ", 100, 24)
        self.assertEqual(len(res.keyframes), 0)

    def test_auto_tag_without_keyframes(self):
        res = self.parser.parse("{auto}", 100, 24)
        self.assertTrue(res.auto_fill)
        self.assertEqual(len(res.keyframes), 1)
        self.assertEqual(res.keyframes[1].frame, 0)
        self.assertEqual(res.keyframes[1].prompt, "")
        self.assertEqual(res.keyframes[1].negative_prompt, "")
        self.assertEqual(res.keyframes[1].weight, 1.0)

    def test_auto_tag_with_keyframes(self):
        res = self.parser.parse("{auto}\n[10] hello", 100, 24)
        self.assertTrue(res.auto_fill)
        self.assertEqual(len(res.keyframes), 1)
        self.assertEqual(res.keyframes[1].frame, 10)
        self.assertEqual(res.keyframes[1].prompt, "hello")

    def test_time_formats(self):
        dsl = '''
        [10] absolute
        [50%] percent
        [2s] seconds
        '''
        res = self.parser.parse(dsl, 100, 24)
        self.assertEqual(len(res.keyframes), 3)
        self.assertEqual(res.keyframes[1].frame, 10)
        self.assertEqual(res.keyframes[2].frame, 48)  # 2s * 24fps = 48
        self.assertEqual(res.keyframes[3].frame, 50)  # 50% of 100 = 50

    def test_negative_prompts(self):
        dsl = '''
        [0] a beautiful cat
        -- ugly, blurry
        -- worst quality
        '''
        res = self.parser.parse(dsl, 100, 24)
        self.assertEqual(len(res.keyframes), 1)
        self.assertEqual(res.keyframes[1].prompt, "a beautiful cat")
        self.assertEqual(res.keyframes[1].negative_prompt, "ugly, blurry worst quality")

    def test_weights_and_transitions(self):
        dsl = '''
        [0]
        weight: 1.5
        transition: blend
        blend: 12
        a glowing orb
        '''
        res = self.parser.parse(dsl, 100, 24)
        self.assertEqual(len(res.keyframes), 1)
        kf = res.keyframes[1]
        self.assertEqual(kf.weight, 1.5)
        self.assertEqual(kf.transition, "blend")
        self.assertEqual(kf.transition_frames, 12)
        self.assertEqual(kf.prompt, "a glowing orb")

    def test_w_shortcut(self):
        dsl = '''
        [0]
        w: 0.8
        test
        '''
        res = self.parser.parse(dsl, 100, 24)
        self.assertEqual(res.keyframes[1].weight, 0.8)

    def test_file_redirect_missing(self):
        # A missing file will trigger warning and return empty
        res = self.parser.parse("file: missing_file.txt", 100, 24)
        self.assertEqual(len(res.keyframes), 0)

if __name__ == '__main__':
    unittest.main()
