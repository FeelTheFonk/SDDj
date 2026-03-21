--
-- PixyToon — Utility Functions
--

return function(PT)

-- ─── Temp File Management ───────────────────────────────────

function PT.get_tmp_dir()
  return app.fs.tempPath or os.getenv("TEMP") or os.getenv("TMP") or "."
end

function PT.make_tmp_path(prefix)
  PT.state.file_counter = PT.state.file_counter + 1
  return app.fs.joinPath(PT.get_tmp_dir(),
    "pixytoon_" .. prefix .. "_" .. PT.state.session_id .. "_" .. PT.state.file_counter .. ".png")
end

-- ─── Image I/O ──────────────────────────────────────────────

function PT.image_to_base64(img)
  local tmp = PT.make_tmp_path("b64")
  if not img:saveAs(tmp) then return nil end
  local f = io.open(tmp, "rb")
  if not f then return nil end
  local data = f:read("*a")
  f:close()
  os.remove(tmp)
  return PT.base64_encode(data)
end

-- Raw PNG bytes for binary WebSocket send (skips base64 overhead).
function PT.image_to_png_bytes(img)
  local tmp = PT.make_tmp_path("bin")
  if not img:saveAs(tmp) then return nil end
  local f = io.open(tmp, "rb")
  if not f then return nil end
  local data = f:read("*a")
  f:close()
  os.remove(tmp)
  return data
end

-- ─── Timer Lifecycle ────────────────────────────────────────

-- Stop a timer safely. Returns nil for idiomatic reassignment:
--   timer = PT.stop_timer(timer)
function PT.stop_timer(t)
  if t then
    if t.isRunning then t:stop() end
  end
  return nil
end

end
