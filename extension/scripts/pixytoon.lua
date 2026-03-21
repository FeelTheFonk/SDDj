--
-- PixyToon — Aseprite Extension for AI Pixel Art Generation
--
-- Connects to the local PixyToon Python server via WebSocket
-- and provides a full GUI for generating pixel art sprites.
--
-- Architecture: shared context table (PT) loaded by sub-modules via dofile().
-- Module load order matters: later modules reference functions defined by earlier ones.
-- All cross-module calls resolve at runtime (not load time), so circular references work.
--
-- Uses the official Aseprite plugin pattern (init/exit) for reliable path resolution.
--

-- ─── JSON Loader (standalone, loaded before PT) ───────────

local function load_json(scripts_dir)
  local json_path = app.fs.joinPath(scripts_dir, "json.lua")
  if not app.fs.isFile(json_path) then
    app.alert("PixyToon: json.lua not found in:\n" .. scripts_dir)
    return nil
  end
  local ok, result = pcall(dofile, json_path)
  if not ok or not result then
    app.alert("PixyToon: Failed to load json.lua\n" .. tostring(result))
    return nil
  end
  return result
end

-- ─── Module Loader ────────────────────────────────────────

local function load_module(PT, scripts_dir, name)
  local path = app.fs.joinPath(scripts_dir, name .. ".lua")
  if not app.fs.isFile(path) then
    app.alert("PixyToon: Module not found: " .. name .. "\nLooked in: " .. scripts_dir)
    return false
  end
  local ok, init_fn = pcall(dofile, path)
  if not ok then
    app.alert("PixyToon: Failed to load " .. name .. "\n" .. tostring(init_fn))
    return false
  end
  if type(init_fn) ~= "function" then
    app.alert("PixyToon: Module " .. name .. " did not return an init function"
      .. "\nGot: " .. type(init_fn) .. " = " .. tostring(init_fn))
    return false
  end
  local init_ok, init_err = pcall(init_fn, PT)
  if not init_ok then
    app.alert("PixyToon: Module " .. name .. " init failed\n" .. tostring(init_err))
    return false
  end
  return true
end

-- ─── Plugin Lifecycle ─────────────────────────────────────

local PT  -- shared context, accessible by exit()

function init(plugin)
  -- plugin.path is the extension root (where package.json lives),
  -- guaranteed by Aseprite — no debug.getinfo hacks needed.
  local scripts_dir = app.fs.joinPath(plugin.path, "scripts")

  -- Validate that the scripts directory actually exists
  if not app.fs.isDirectory(scripts_dir) then
    app.alert("PixyToon: Scripts directory not found:\n" .. scripts_dir)
    return
  end

  -- Load JSON library
  local json = load_json(scripts_dir)
  if not json then return end

  -- Create shared context table
  PT = { json = json, plugin = plugin }

  -- Load modules in dependency order
  local modules = {
    "pixytoon_base64",    -- pure codec, no deps
    "pixytoon_state",     -- constants + state tables
    "pixytoon_utils",     -- temp files, image I/O, timer helper
    "pixytoon_settings",  -- save/load/apply
    "pixytoon_ws",        -- WebSocket transport + connection
    "pixytoon_capture",   -- image capture (active layer, flattened, mask)
    "pixytoon_request",   -- request builders (parse, attach, build)
    "pixytoon_import",    -- import result, animation frame, live preview
    "pixytoon_live",      -- live paint system (hash, dirty region, timers)
    "pixytoon_handler",   -- response dispatch table
    "pixytoon_dialog",    -- dialog construction (tabs + actions)
  }

  for _, name in ipairs(modules) do
    if not load_module(PT, scripts_dir, name) then return end
  end

  -- Launch the dialog
  PT.build_dialog()
  PT.apply_settings(PT.load_settings())
end

function exit(plugin)
  if not PT then return end

  -- Stop all timers
  if PT.timers then
    for key, timer in pairs(PT.timers) do
      if timer then pcall(function() timer:stop() end) end
      PT.timers[key] = nil
    end
  end

  -- Stop live mode timers
  if PT.stop_live_timer then pcall(PT.stop_live_timer) end

  -- Disconnect WebSocket
  if PT.ws_handle then
    pcall(function() PT.ws_handle:close() end)
    PT.ws_handle = nil
  end

  -- Save settings before exit
  if PT.save_settings then pcall(PT.save_settings) end
end
