--
-- SDDj — Aseprite Extension for SD Generation & Animation
--
-- Connects to the local SDDj Python server via WebSocket
-- and provides a full GUI for generating images and animations.
--
-- Architecture: shared context table (PT) loaded by sub-modules via dofile().
-- Module load order matters: later modules reference functions defined by earlier ones.
-- All cross-module calls resolve at runtime (not load time), so circular references work.
--
-- Follows the standard Aseprite multi-file extension pattern:
--   1. Top-level dofile("./module.lua") for loading (relative paths work here)
--   2. init(plugin) for plugin-specific setup (dialog, settings)
--   3. exit(plugin) for cleanup (timers, WebSocket, settings)
--

-- ─── JSON Loader ──────────────────────────────────────────

local json_ok, json = pcall(dofile, "./json.lua")
if not json_ok or not json then
  app.alert("SDDj: Failed to load json.lua\n" .. tostring(json))
  return
end

local parser_ok, dsl_parser = pcall(dofile, "./sddj_dsl_parser.lua")
if not parser_ok or type(dsl_parser) ~= "table" then
  app.alert("SDDj: Failed to load sddj_dsl_parser.lua\n" .. tostring(dsl_parser))
  return
end

-- ─── Shared Context ───────────────────────────────────────

local _PT = { json = json, dsl_parser = dsl_parser }

-- ─── Module Loader ────────────────────────────────────────
-- dofile("./name.lua") resolves relative to the calling script's
-- directory (Aseprite's custom dofile uses a current_script_dirs stack).
-- This only works at the top level, while the file is being executed.

local modules = {
  "sddj_base64",    -- pure codec, no deps
  "sddj_state",     -- constants + state tables
  "sddj_utils",     -- temp files, image I/O, timer helper, deep copy
  "sddj_settings",  -- save/load/apply
  "sddj_ws",        -- WebSocket transport + connection
  "sddj_capture",   -- image capture (active layer, flattened, mask)
  "sddj_request",   -- request builders (parse, attach, build)
  "sddj_dsl_editor",-- schedule editor popup, timeline, presets
  "sddj_import",    -- import result, animation frame
  "sddj_output",    -- output directory, metadata persistence, load/apply
  "sddj_handler",   -- response dispatch table
  "sddj_dialog",    -- dialog construction (tabs + actions)
}

for _, name in ipairs(modules) do
  local ok, init_fn = pcall(dofile, "./" .. name .. ".lua")
  if not ok then
    app.alert("SDDj: Failed to load " .. name .. "\n" .. tostring(init_fn))
    return
  end
  if type(init_fn) ~= "function" then
    app.alert("SDDj: Module " .. name .. " did not return an init function"
      .. "\nGot: " .. type(init_fn) .. " = " .. tostring(init_fn))
    return
  end
  local init_ok, init_err = pcall(init_fn, _PT)
  if not init_ok then
    app.alert("SDDj: Module " .. name .. " init failed\n" .. tostring(init_err))
    return
  end
end

-- ─── Plugin Lifecycle ─────────────────────────────────────
-- Aseprite calls init(plugin) after executing this file.
-- By this point, all modules are loaded and all functions in _PT are ready.

function init(plugin)
  -- Clean up any leftover temp files from previous sessions
  if _PT.cleanup_session_temp_files then
    pcall(_PT.cleanup_session_temp_files, true)  -- all_sessions = true
  end

  _PT.build_dialog()
  _PT.apply_settings(_PT.load_settings())
end

function exit(plugin)
  if not _PT then return end
  local pt = _PT
  _PT = nil  -- prevent re-entry from nested event pumping

  -- 1. Stop ALL timers FIRST (prevent callbacks during teardown)
  if pt.timers then
    for key, timer in pairs(pt.timers) do
      if timer then pcall(function() timer:stop() end) end
      pt.timers[key] = nil
    end
  end

  -- 1b. Stop module-private timers (not in PT.timers table)
  -- _refresh_timer (30fps app.refresh) and _drain_timer (1ms response drain)
  -- are local variables in sddj_handler.lua, only reachable via these functions.
  if pt.stop_refresh_timer then pcall(pt.stop_refresh_timer) end
  if pt.clear_response_queue then pcall(pt.clear_response_queue) end

  -- 2. Disarm connection state (prevents heartbeat/reconnect callbacks)
  if pt.state then
    pt.state.connected = false
    pt.state.connecting = false
  end
  if pt.reconnect then
    pt.reconnect.manual_disconnect = true
  end

  -- 4. Save settings (primary path via dlg, fallback via cached JSON)
  if pt.save_settings then pcall(pt.save_settings) end
  if not pt.dlg and pt._last_encoded_settings and pt.cfg then
    pcall(function()
      -- Atomic fallback: .tmp then rename (matches save_settings pattern)
      local tmp = pt.cfg.SETTINGS_FILE .. ".tmp"
      local f = io.open(tmp, "w")
      if f then
        f:write(pt._last_encoded_settings); f:close()
        pcall(os.remove, pt.cfg.SETTINGS_FILE)
        local rename_ok, rename_err = os.rename(tmp, pt.cfg.SETTINGS_FILE)
        if not rename_ok then
          -- Fallback: direct write
          local f2 = io.open(pt.cfg.SETTINGS_FILE, "w")
          if f2 then
            f2:write(pt._last_encoded_settings)
            f2:close()
          end
        end
      end
    end)
  end

  -- 5. (removed) — NO sendText(shutdown) or sendText(cancel) at exit.
  -- Server detects client disconnect via TCP FIN/RST (server.py l.266-269).
  -- start.ps1 handles graceful shutdown via HTTP POST /shutdown.
  -- sendText() can block indefinitely on Windows Winsock if TCP buffer full
  -- or server unreachable — this was the root cause of the Aseprite hang.

  -- 6. Abandon WebSocket — do NOT call close() (may block indefinitely)
  -- OS tears down TCP socket on process exit; server detects disconnect.
  pt.ws_handle = nil

  -- 7. Clean up session temp files
  if pt.cleanup_session_temp_files then
    pcall(pt.cleanup_session_temp_files)
  end
end
