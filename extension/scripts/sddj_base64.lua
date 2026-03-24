--
-- SDDj — Base64 Codec (optimized)
--

return function(PT)

local b64chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

-- ─── Encode (unchanged — encode is fast enough) ─────────────

function PT.base64_encode(data)
  return ((data:gsub(".", function(x)
    local r, b = "", x:byte()
    for i = 8, 1, -1 do r = r .. (b % 2^i - b % 2^(i-1) > 0 and "1" or "0") end
    return r
  end) .. "0000"):gsub("%d%d%d?%d?%d?%d?", function(x)
    if #x < 6 then return "" end
    local c = 0
    for i = 1, 6 do c = c + (x:sub(i,i) == "1" and 2^(6-i) or 0) end
    return b64chars:sub(c+1, c+1)
  end) .. ({ "", "==", "=" })[#data % 3 + 1])
end

-- ─── Decode (lookup table — O(1) per character) ─────────────
-- Previous: string:find() per character = O(32) per lookup
-- Now: byte-indexed lookup table = O(1) per lookup
-- + table.concat instead of string concatenation (reduces GC pressure)
-- + math.floor(acc / 2^bits) for O(1) byte extraction (no inner loop)

local _b64_lut = {}
for i = 1, #b64chars do
  _b64_lut[b64chars:byte(i)] = i - 1
end

-- Pre-computed power-of-two table to avoid repeated 2^n calls
local _pow2 = {}
for i = 0, 24 do _pow2[i] = 2 ^ i end

function PT.base64_decode(data)
  local t, n = {}, 0
  local acc, bits = 0, 0
  for i = 1, #data do
    local v = _b64_lut[data:byte(i)]
    if v then
      acc = acc * 64 + v
      bits = bits + 6
      if bits >= 8 then
        bits = bits - 8
        n = n + 1
        t[n] = string.char(math.floor(acc / _pow2[bits]) % 256)
        acc = acc % _pow2[bits]
      end
    end
  end
  return table.concat(t)
end

end
