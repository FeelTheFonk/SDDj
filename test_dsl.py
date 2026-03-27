import lupa
lua = lupa.LuaRuntime()
print("Matches empty string:", lua.eval('string.match("", "^%s*$") ~= nil'))
print("Matches spaces:", lua.eval('string.match("   ", "^%s*$") ~= nil'))
print("Matches newline:", lua.eval('string.match("\\n", "^%s*$") ~= nil'))
print("Returns value for empty:", repr(lua.eval('string.match("", "^%s*$")')))
