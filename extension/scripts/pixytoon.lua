--
-- PixyToon — Aseprite Extension for AI Pixel Art Generation
--
-- Connects to the local PixyToon Python server via WebSocket
-- and provides a full GUI for generating pixel art sprites.
--

-- ─── JSON LOADER (robust) ───────────────────────────────────

local json
do
  local scripts_path = app.fs.joinPath(app.fs.userConfigPath, "scripts", "json.lua")
  local ext_path = app.fs.joinPath(app.fs.userConfigPath, "extensions", "pixytoon", "scripts", "json.lua")
  local load_ok, load_result
  if app.fs.isFile(scripts_path) then
    load_ok, load_result = pcall(dofile, scripts_path)
  elseif app.fs.isFile(ext_path) then
    load_ok, load_result = pcall(dofile, ext_path)
  else
    load_ok = false
    load_result = "json.lua not found at:\n" .. scripts_path .. "\n" .. ext_path
  end
  if not load_ok or not load_result then
    app.alert("PixyToon: Failed to load json.lua\n" .. tostring(load_result))
    return
  end
  json = load_result
end

-- ─── BASE64 ──────────────────────────────────────────────────

local b64chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

local function base64_encode(data)
  return ((data:gsub('.', function(x)
    local r, b = '', x:byte()
    for i = 8, 1, -1 do r = r .. (b % 2^i - b % 2^(i-1) > 0 and '1' or '0') end
    return r
  end) .. '0000'):gsub('%d%d%d?%d?%d?%d?', function(x)
    if #x < 6 then return '' end
    local c = 0
    for i = 1, 6 do c = c + (x:sub(i,i) == '1' and 2^(6-i) or 0) end
    return b64chars:sub(c+1, c+1)
  end) .. ({ '', '==', '=' })[#data % 3 + 1])
end

local function base64_decode(data)
  data = data:gsub('[^' .. b64chars .. '=]', '')
  return (data:gsub('.', function(x)
    if x == '=' then return '' end
    local r, f = '', (b64chars:find(x) - 1)
    for i = 6, 1, -1 do r = r .. (f % 2^i - f % 2^(i-1) > 0 and '1' or '0') end
    return r
  end):gsub('%d%d%d?%d?%d?%d?%d?%d?', function(x)
    if #x ~= 8 then return '' end
    local c = 0
    for i = 1, 8 do c = c + (x:sub(i,i) == '1' and 2^(8-i) or 0) end
    return string.char(c)
  end))
end

-- ─── STATE ───────────────────────────────────────────────────

local SERVER_URL = "ws://127.0.0.1:9876/ws"
local ws = nil
local dlg = nil
local connected = false
local generating = false
local available_palettes = {}
local resources_requested = false
local connect_timer = nil
local heartbeat_timer = nil
local gen_step_start = nil
local _file_counter = 0
local _session_id = tostring(os.time()) .. "_" .. tostring(math.random(1000, 9999))

-- Animation state
local animating = false
local anim_layer = nil
local anim_start_frame = 0
local anim_frame_count = 0
local anim_base_seed = 0
local available_loras = {}
local available_embeddings = {}

-- Live paint state
local live_mode = false
local live_timer = nil
local live_canvas_hash = nil
local live_frame_id = 0
local live_request_inflight = false
local live_preview_layer = nil
local live_last_prompt = nil

-- Forward declarations
local handle_response
local import_result
local import_animation_frame
local request_resources
local start_heartbeat
local stop_heartbeat
local stop_live_timer
local start_live_timer
local live_update_preview

-- ─── HELPERS ─────────────────────────────────────────────────

local function get_tmp_dir()
  return app.fs.tempPath or os.getenv("TEMP") or os.getenv("TMP") or "."
end

local function make_tmp_path(prefix)
  _file_counter = _file_counter + 1
  return app.fs.joinPath(get_tmp_dir(),
    "pixytoon_" .. prefix .. "_" .. _session_id .. "_" .. _file_counter .. ".png")
end

local function image_to_base64(img)
  local tmp = make_tmp_path("b64")
  img:saveAs(tmp)
  local f = io.open(tmp, "rb")
  if not f then return nil end
  local data = f:read("*a")
  f:close()
  os.remove(tmp)
  return base64_encode(data)
end

local function build_post_process()
  local pp = {
    pixelate = {
      enabled = dlg.data.pixelate,
      target_size = dlg.data.pixel_size
    },
    quantize_method = dlg.data.quantize_method,
    quantize_colors = dlg.data.colors,
    dither = dlg.data.dither,
    palette = { mode = dlg.data.palette_mode },
    remove_bg = dlg.data.remove_bg
  }
  if dlg.data.palette_mode == "preset" then
    pp.palette.name = dlg.data.palette_name
  elseif dlg.data.palette_mode == "custom" then
    local hex_str = dlg.data.palette_custom_colors or ""
    local colors = {}
    for hex in hex_str:gmatch("#?(%x%x%x%x%x%x)") do
      colors[#colors + 1] = "#" .. hex
    end
    if #colors > 0 then pp.palette.colors = colors end
  end
  return pp
end

-- ─── WEBSOCKET ───────────────────────────────────────────────

local function update_status(text)
  if dlg then dlg:modify{ id = "status", text = text } end
end

local function set_connected(state)
  connected = state
  if state then start_heartbeat() else stop_heartbeat() end
  if not dlg then return end
  if state then
    dlg:modify{ id = "connect_btn", text = "Disconnect" }
    dlg:modify{ id = "generate_btn", enabled = true }
    dlg:modify{ id = "animate_btn", enabled = true }
    dlg:modify{ id = "live_btn", enabled = true }
  else
    dlg:modify{ id = "connect_btn", text = "Connect" }
    dlg:modify{ id = "generate_btn", enabled = false }
    dlg:modify{ id = "cancel_btn", enabled = false }
    dlg:modify{ id = "animate_btn", enabled = false }
    dlg:modify{ id = "live_btn", enabled = false }
    dlg:modify{ id = "live_btn", text = "START LIVE" }
    dlg:modify{ id = "live_accept_btn", visible = false }
    if generating then generating = false end
    if animating then animating = false end
    stop_live_timer()
    live_mode = false
    live_request_inflight = false
  end
end

local function stop_connect_timer()
  if connect_timer then
    if connect_timer.isRunning then connect_timer:stop() end
    connect_timer = nil
  end
end

stop_heartbeat = function()
  if heartbeat_timer then
    if heartbeat_timer.isRunning then heartbeat_timer:stop() end
    heartbeat_timer = nil
  end
end

start_heartbeat = function()
  stop_heartbeat()
  heartbeat_timer = Timer{
    interval = 30.0,
    ontick = function()
      if connected and ws and not generating and not animating and not live_mode then
        pcall(function() ws:sendText('{"action":"ping"}') end)
      end
    end,
  }
  heartbeat_timer:start()
end

local function connect()
  if ws then pcall(function() ws:close() end); ws = nil end
  update_status("Connecting...")
  ws = WebSocket{
    url = SERVER_URL,
    onreceive = function(msg_type, data)
      if msg_type == WebSocketMessageType.OPEN then
        stop_connect_timer()
        set_connected(true)
        update_status("Connected")
        pcall(function() ws:sendText(json.encode({ action = "ping" })) end)
        return
      end
      if msg_type == WebSocketMessageType.CLOSE then
        set_connected(false)
        resources_requested = false
        update_status("Disconnected (server closed)")
        ws = nil
        return
      end
      if msg_type == WebSocketMessageType.TEXT then
        if not connected then
          stop_connect_timer()
          set_connected(true)
          update_status("Connected")
        end
        local ok, response = pcall(json.decode, data)
        if not ok then return end
        local hok, herr = pcall(handle_response, response)
        if not hok then update_status("Error: " .. tostring(herr)) end
      end
    end,
    deflate = false,
  }
  ws:connect()

  -- Connection timeout
  stop_connect_timer()
  connect_timer = Timer{
    interval = 5.0,
    ontick = function()
      stop_connect_timer()
      if not connected then
        if ws then pcall(function() ws:close() end); ws = nil end
        update_status("Connection failed - is the server running?")
      end
    end,
  }
  connect_timer:start()
end

local function disconnect()
  stop_connect_timer()
  if ws then pcall(function() ws:close() end); ws = nil end
  set_connected(false)
  resources_requested = false
  anim_layer = nil
  anim_start_frame = 0
  anim_frame_count = 0
  anim_base_seed = 0
  generating = false
  animating = false
  stop_live_timer()
  live_mode = false
  live_request_inflight = false
  live_canvas_hash = nil
  live_preview_layer = nil
  update_status("Disconnected")
end

local function send(payload)
  if not connected or ws == nil then
    update_status("Not connected")
    return false
  end
  local ok, err = pcall(function() ws:sendText(json.encode(payload)) end)
  if not ok then
    update_status("Send failed: " .. tostring(err))
    return false
  end
  return true
end

request_resources = function()
  resources_requested = true
  send({ action = "list_palettes" })
  send({ action = "list_loras" })
  send({ action = "list_embeddings" })
end

-- ─── RESPONSE HANDLER ────────────────────────────────────────

handle_response = function(resp)
  if resp.type == "progress" then
    if not resp.total or resp.total <= 0 then return end
    if not dlg then return end
    local pct = math.floor((resp.step / resp.total) * 100)
    local eta_str = ""
    local now = os.clock()
    if gen_step_start and resp.step > 1 then
      local elapsed = now - gen_step_start
      local steps_done = resp.step - 1
      if steps_done > 0 then
        local remaining = (elapsed / steps_done) * (resp.total - resp.step)
        if remaining < 60 then
          eta_str = string.format(" ~%.0fs", remaining)
        else
          eta_str = string.format(" ~%.1fmin", remaining / 60)
        end
      end
    end
    local frame_ctx = ""
    if resp.frame_index ~= nil and resp.total_frames ~= nil then
      frame_ctx = " [F" .. (resp.frame_index + 1) .. "/" .. resp.total_frames .. "]"
    end
    update_status(resp.step .. "/" .. resp.total .. " (" .. pct .. "%)" .. frame_ctx .. eta_str)

  elseif resp.type == "result" then
    generating = false
    gen_step_start = nil
    if dlg then
      update_status("Done (" .. tostring(resp.time_ms or "?") .. "ms, seed=" .. tostring(resp.seed or "?") .. ")")
      dlg:modify{ id = "generate_btn", enabled = true }
      dlg:modify{ id = "cancel_btn", enabled = false }
    end
    if resp.image then import_result(resp) end

  elseif resp.type == "animation_frame" then
    if resp.image and resp.frame_index ~= nil then
      import_animation_frame(resp)
    end

  elseif resp.type == "animation_complete" then
    animating = false
    gen_step_start = nil
    if dlg then
      local tag_str = ""
      if resp.tag_name and resp.tag_name ~= "" then tag_str = ", tag=" .. resp.tag_name end
      update_status("Animation done (" .. tostring(resp.total_frames or "?") .. " frames, "
        .. tostring(resp.total_time_ms or "?") .. "ms" .. tag_str .. ")")
      dlg:modify{ id = "animate_btn", enabled = true }
      dlg:modify{ id = "cancel_btn", enabled = false }
    end

    local spr = app.sprite
    if spr and anim_frame_count > 0 then
      local dur = (dlg and dlg.data.anim_duration or 100) / 1000.0
      for i = 0, anim_frame_count - 1 do
        local fn = anim_start_frame + i
        if spr.frames[fn] then spr.frames[fn].duration = dur end
      end
      local tag_start = anim_start_frame
      local tag_end = anim_start_frame + anim_frame_count - 1
      if resp.tag_name and resp.tag_name ~= "" and spr.frames[tag_start] and spr.frames[tag_end] then
        local tag = spr:newTag(tag_start, tag_end)
        tag.name = resp.tag_name
      end
      app.refresh()
    end
    anim_layer = nil
    anim_start_frame = 0
    anim_frame_count = 0
    anim_base_seed = 0

  elseif resp.type == "error" then
    local was_animating = animating
    generating = false
    animating = false
    gen_step_start = nil

    if was_animating and anim_frame_count > 0 then
      local spr = app.sprite
      if spr then
        local dur = (dlg and dlg.data.anim_duration or 100) / 1000.0
        for i = 0, anim_frame_count - 1 do
          local fn = anim_start_frame + i
          if spr.frames[fn] then spr.frames[fn].duration = dur end
        end
      end
      anim_layer = nil
      anim_start_frame = 0
      anim_frame_count = 0
      anim_base_seed = 0
    end

    if dlg then
      update_status("Error: " .. tostring(resp.message or "Unknown"))
      dlg:modify{ id = "generate_btn", enabled = not live_mode }
      dlg:modify{ id = "animate_btn", enabled = not live_mode }
      dlg:modify{ id = "cancel_btn", enabled = false }
      live_request_inflight = false
    end
    if resp.code ~= "CANCELLED" then
      app.alert("PixyToon: " .. tostring(resp.message or "Unknown error"))
    end

  elseif resp.type == "list" then
    local lt = resp.list_type or ""
    local items = resp.items or {}
    if lt == "palettes" then
      available_palettes = items
      if dlg and #items > 0 then
        local opts = {}
        for _, n in ipairs(items) do opts[#opts + 1] = n end
        dlg:modify{ id = "palette_name", options = opts }
      end
    elseif lt == "loras" then
      available_loras = items
      if dlg then
        local opts = { "(default)" }
        for _, n in ipairs(items) do opts[#opts + 1] = n end
        dlg:modify{ id = "lora_name", options = opts }
      end
    elseif lt == "embeddings" then
      available_embeddings = items
    end
    local total = #available_palettes + #available_loras + #available_embeddings
    if total > 0 then
      update_status("Resources loaded (" .. #available_loras .. " LoRAs, "
        .. #available_palettes .. " palettes, " .. #available_embeddings .. " embeddings)")
    else
      update_status("Connected (no resources found)")
    end

  elseif resp.type == "realtime_ready" then
    live_mode = true
    live_request_inflight = false
    live_last_prompt = dlg and dlg.data.prompt or nil
    if dlg then
      update_status("Live mode active")
      dlg:modify{ id = "live_btn", text = "STOP LIVE" }
      dlg:modify{ id = "live_accept_btn", visible = true }
      dlg:modify{ id = "generate_btn", enabled = false }
      dlg:modify{ id = "animate_btn", enabled = false }
    end
    start_live_timer()

  elseif resp.type == "realtime_result" then
    live_request_inflight = false
    if live_mode and resp.image then
      live_update_preview(resp)
      if dlg then
        update_status("Live (" .. tostring(resp.latency_ms or "?") .. "ms)")
      end
    end

  elseif resp.type == "realtime_stopped" then
    stop_live_timer()
    live_mode = false
    live_request_inflight = false
    live_canvas_hash = nil
    if dlg then
      update_status("Live mode stopped")
      dlg:modify{ id = "live_btn", text = "START LIVE" }
      dlg:modify{ id = "live_accept_btn", visible = false }
      dlg:modify{ id = "generate_btn", enabled = true }
      dlg:modify{ id = "animate_btn", enabled = true }
    end

  elseif resp.type == "pong" then
    if not connected then set_connected(true) end
    if not resources_requested then request_resources() end
    update_status("Connected")
  end
end

-- ─── IMPORT RESULT ───────────────────────────────────────────

import_result = function(resp)
  local img_data = base64_decode(resp.image)
  local tmp = make_tmp_path("res")

  local ok, err = pcall(function()
    local f = io.open(tmp, "wb")
    if not f then error("Failed to create temp file") end
    f:write(img_data)
    f:close()

    local spr = app.sprite
    if spr == nil then
      spr = Sprite(resp.width or 512, resp.height or 512, ColorMode.RGB)
    end

    local layer = spr:newLayer()
    layer.name = "PixyToon #" .. tostring(resp.seed or "?")

    local img = Image{ fromFile = tmp }
    if img then spr:newCel(layer, app.frame, img, Point(0, 0)) end

    os.remove(tmp)
    app.refresh()
  end)
  if not ok then
    pcall(os.remove, tmp)
    update_status("Import error: " .. tostring(err))
  end
end

-- ─── IMPORT ANIMATION FRAME ─────────────────────────────────

import_animation_frame = function(resp)
  if not animating then return end
  if resp.frame_index ~= 0 and anim_layer == nil then return end

  local img_data = base64_decode(resp.image)
  local tmp = make_tmp_path("anim")

  local ok, err = pcall(function()
    local f = io.open(tmp, "wb")
    if not f then return end
    f:write(img_data)
    f:close()

    local spr = app.sprite
    local created_sprite = false
    if spr == nil then
      spr = Sprite(resp.width or 512, resp.height or 512, ColorMode.RGB)
      created_sprite = true
    end

    -- First frame: create layer and anchor position
    if resp.frame_index == 0 then
      anim_layer = spr:newLayer()
      anim_layer.name = "PixyToon Anim #" .. tostring(resp.seed or "?")
      anim_base_seed = resp.seed or 0
      anim_frame_count = 0
      if created_sprite then
        anim_start_frame = 1  -- reuse the initial empty frame
      else
        anim_start_frame = #spr.frames + 1  -- append after existing content
      end
    end

    -- Determine frame position
    local frame_num
    if resp.frame_index == 0 and created_sprite then
      frame_num = 1
    else
      local target_pos = anim_start_frame + resp.frame_index
      if target_pos > #spr.frames + 1 then
        target_pos = #spr.frames + 1
      end
      local new_frame = spr:newEmptyFrame(target_pos)
      frame_num = new_frame.frameNumber
    end

    local img = Image{ fromFile = tmp }
    if img and anim_layer and spr.frames[frame_num] then
      spr:newCel(anim_layer, spr.frames[frame_num], img, Point(0, 0))
    end

    anim_frame_count = anim_frame_count + 1
    os.remove(tmp)
    app.refresh()

    if dlg then
      update_status("Frame " .. (resp.frame_index + 1) .. "/" .. tostring(resp.total_frames or "?")
        .. " (" .. tostring(resp.time_ms or "?") .. "ms)")
    end
  end)
  if not ok then
    pcall(os.remove, tmp)
    update_status("Import error: " .. tostring(err))
  end
end

-- ─── CAPTURE FUNCTIONS ──────────────────────────────────────

local function capture_active_layer()
  local spr = app.sprite
  if spr == nil then return nil end
  local cel = app.cel
  if cel == nil or cel.image == nil then return nil end
  local full = Image(spr.spec)
  full:clear()
  full:drawImage(cel.image, cel.position)
  return image_to_base64(full)
end

local function capture_flattened()
  local spr = app.sprite
  if spr == nil then return nil end
  local flat_img = Image(spr.spec)
  flat_img:drawSprite(spr, app.frame)
  return image_to_base64(flat_img)
end

local function capture_mask()
  local spr = app.sprite
  if spr == nil then return nil end

  -- Strategy A: active selection
  local sel = spr.selection
  if sel and not sel.isEmpty then
    local mask_img = Image(spr.width, spr.height, ColorMode.GRAY)
    mask_img:clear(Color{ gray = 0 })
    for y = sel.bounds.y, sel.bounds.y + sel.bounds.height - 1 do
      for x = sel.bounds.x, sel.bounds.x + sel.bounds.width - 1 do
        if sel:contains(x, y) then
          mask_img:drawPixel(x, y, Color{ gray = 255 })
        end
      end
    end
    return image_to_base64(mask_img)
  end

  -- Strategy B: "Mask" layer (with correct positioning)
  local function find_mask_layer(layers)
    for _, layer in ipairs(layers) do
      if layer.name == "Mask" or layer.name == "mask" then
        local cel = layer:cel(app.frame)
        if cel and cel.image then
          local full = Image(spr.width, spr.height, ColorMode.GRAY)
          full:clear(Color{ gray = 0 })
          full:drawImage(cel.image, cel.position)
          return image_to_base64(full)
        end
      end
      if layer.isGroup and layer.layers then
        local result = find_mask_layer(layer.layers)
        if result then return result end
      end
    end
    return nil
  end
  local mask_b64 = find_mask_layer(spr.layers)
  if mask_b64 then return mask_b64 end

  -- Strategy C: auto from active layer alpha
  local cel = app.cel
  if cel and cel.image then
    local img = cel.image
    local mask_img = Image(spr.width, spr.height, ColorMode.GRAY)
    mask_img:clear(Color{ gray = 0 })
    local ox, oy = cel.position.x, cel.position.y
    for y = 0, img.height - 1 do
      for x = 0, img.width - 1 do
        local px = img:getPixel(x, y)
        local a = app.pixelColor.rgbaA(px)
        if a > 0 then
          local sx, sy = ox + x, oy + y
          if sx >= 0 and sx < spr.width and sy >= 0 and sy < spr.height then
            mask_img:drawPixel(sx, sy, Color{ gray = 255 })
          end
        end
      end
    end
    return image_to_base64(mask_img)
  end

  return nil
end

-- ─── LIVE PAINT HELPERS ─────────────────────────────────────

local function canvas_hash(img)
  local w, h = img.width, img.height
  local hash = 0
  local step = math.max(1, math.floor(math.min(w, h) / 32))
  for y = 0, h - 1, step do
    for x = 0, w - 1, step do
      hash = (hash * 31 + img:getPixel(x, y)) % 2147483647
    end
  end
  return hash
end

stop_live_timer = function()
  if live_timer then
    if live_timer.isRunning then live_timer:stop() end
    live_timer = nil
  end
end

start_live_timer = function()
  stop_live_timer()
  live_timer = Timer{
    interval = 0.3,
    ontick = function()
      if not live_mode or not connected or live_request_inflight then return end
      local spr = app.sprite
      if spr == nil then return end

      -- Hide preview layer for capture
      local was_visible = true
      if live_preview_layer then
        was_visible = live_preview_layer.isVisible
        live_preview_layer.isVisible = false
      end
      local flat_img = Image(spr.spec)
      flat_img:drawSprite(spr, app.frame)
      if live_preview_layer then
        live_preview_layer.isVisible = was_visible
      end

      -- Check if canvas changed
      local hash = canvas_hash(flat_img)
      if hash == live_canvas_hash then return end
      live_canvas_hash = hash

      -- Auto-detect prompt changes
      if dlg then
        local current_prompt = dlg.data.prompt
        if current_prompt ~= live_last_prompt then
          live_last_prompt = current_prompt
          send({ action = "realtime_update", prompt = current_prompt })
        end
      end

      -- Send frame
      local b64 = image_to_base64(flat_img)
      if not b64 then return end
      live_frame_id = live_frame_id + 1
      live_request_inflight = true
      send({
        action = "realtime_frame",
        image = b64,
        frame_id = live_frame_id,
      })
    end,
  }
  live_timer:start()
end

live_update_preview = function(resp)
  local spr = app.sprite
  if spr == nil then return end

  -- Find or create preview layer
  if live_preview_layer == nil or not pcall(function() return live_preview_layer.name end) then
    live_preview_layer = nil
    for _, layer in ipairs(spr.layers) do
      if layer.name == "_pixytoon_live" then
        live_preview_layer = layer
        break
      end
    end
    if live_preview_layer == nil then
      live_preview_layer = spr:newLayer()
      live_preview_layer.name = "_pixytoon_live"
    end
  end

  local img_data = base64_decode(resp.image)
  local tmp = make_tmp_path("live")
  local f = io.open(tmp, "wb")
  if not f then return end
  f:write(img_data)
  f:close()

  local img = Image{ fromFile = tmp }
  os.remove(tmp)
  if not img then return end

  -- Replace existing cel
  local cel = live_preview_layer:cel(app.frame)
  if cel then spr:deleteCel(cel) end
  spr:newCel(live_preview_layer, app.frame, img, Point(0, 0))

  -- Apply opacity
  if dlg then
    live_preview_layer.opacity = math.floor(dlg.data.live_opacity * 255 / 100)
  end

  app.refresh()
end

-- ─── SHARED REQUEST BUILDERS ─────────────────────────────────

local function parse_size()
  local s = dlg.data.output_size
  local w, h = s:match("(%d+)x(%d+)")
  return tonumber(w), tonumber(h)
end

local function parse_seed()
  local v = tonumber(dlg.data.seed) or -1
  if v ~= math.floor(v) then v = -1 end
  return v
end

local function attach_lora(req)
  local sel = dlg.data.lora_name
  if sel and sel ~= "(default)" then
    req.lora = { name = sel, weight = dlg.data.lora_weight / 100.0 }
  end
end

local function attach_neg_ti(req)
  if dlg.data.use_neg_ti and #available_embeddings > 0 then
    local ti_list = {}
    local w = dlg.data.neg_ti_weight / 100.0
    for _, name in ipairs(available_embeddings) do
      ti_list[#ti_list + 1] = { name = name, weight = w }
    end
    req.negative_ti = ti_list
  end
end

local function attach_source_image(req)
  if req.mode == "img2img" or req.mode:find("controlnet_") then
    local b64 = capture_active_layer()
    if not b64 then
      app.alert("No active layer to use as source.")
      return false
    end
    if req.mode == "img2img" then req.source_image = b64
    else req.control_image = b64 end
  end
  if req.mode == "inpaint" then
    local src = capture_flattened()
    if not src then
      app.alert("Inpaint requires an open sprite.")
      return false
    end
    req.source_image = src
    local mask = capture_mask()
    if not mask then
      app.alert("Inpaint requires a mask.\n- Make a selection, or\n- Create a 'Mask' layer, or\n- Draw on active layer")
      return false
    end
    req.mask_image = mask
  end
  return true
end

-- ─── BUILD DIALOG ────────────────────────────────────────────

local function build_dialog()
  dlg = Dialog{
    title = "PixyToon - AI Pixel Art",
    resizeable = true,
    onclose = function() disconnect() end
  }

  -- ══════════════════════════════════════════════════════════
  -- CONNECTION (always visible)
  -- ══════════════════════════════════════════════════════════
  dlg:separator{ text = "Connection" }

  dlg:entry{
    id = "server_url",
    label = "Server",
    text = SERVER_URL,
    hexpand = true,
  }

  dlg:label{ id = "status", text = "Disconnected" }

  dlg:button{
    id = "connect_btn",
    text = "Connect",
    onclick = function()
      if connected then disconnect()
      else
        SERVER_URL = dlg.data.server_url or SERVER_URL
        connect()
      end
    end
  }
  dlg:button{
    id = "refresh_btn",
    text = "Refresh Resources",
    onclick = function()
      if connected then
        resources_requested = false
        request_resources()
        update_status("Refreshing...")
      else
        update_status("Not connected")
      end
    end
  }

  -- ══════════════════════════════════════════════════════════
  -- TAB: Generate
  -- ══════════════════════════════════════════════════════════
  dlg:tab{ id = "tab_gen", text = "Generate" }

  dlg:combobox{
    id = "mode",
    label = "Mode",
    options = {
      "txt2img", "img2img", "inpaint",
      "controlnet_openpose", "controlnet_canny",
      "controlnet_scribble", "controlnet_lineart",
    },
    option = "txt2img",
  }

  dlg:combobox{
    id = "lora_name",
    label = "LoRA",
    options = { "(default)" },
    option = "(default)",
  }

  dlg:slider{
    id = "lora_weight",
    label = "LoRA (1.00)",
    min = -200,
    max = 200,
    value = 100,
    onchange = function()
      dlg:modify{ id = "lora_weight",
        label = string.format("LoRA (%.2f)", dlg.data.lora_weight / 100.0) }
    end,
  }

  dlg:entry{
    id = "prompt",
    label = "Prompt",
    text = "pixel art, PixArFK, game sprite, sharp pixels",
    hexpand = true,
  }

  dlg:entry{
    id = "negative_prompt",
    label = "Neg. Prompt",
    text = "blurry, antialiased, smooth gradient, photorealistic, 3d render, soft edges, low quality, worst quality",
    hexpand = true,
  }

  dlg:check{
    id = "use_neg_ti",
    label = "Neg. Embeddings",
    selected = false,
    onchange = function()
      dlg:modify{ id = "neg_ti_weight", visible = dlg.data.use_neg_ti }
    end,
  }

  dlg:slider{
    id = "neg_ti_weight",
    label = "Emb. Weight",
    min = 10,
    max = 200,
    value = 100,
    visible = false,
  }

  dlg:combobox{
    id = "output_size",
    label = "Size",
    options = {
      "512x512", "512x768", "768x512", "768x768",
      "384x384", "256x256", "128x128", "64x64",
    },
    option = "512x512",
  }

  dlg:entry{
    id = "seed",
    label = "Seed (-1=rand)",
    text = "-1",
    hexpand = true,
  }

  dlg:slider{
    id = "denoise",
    label = "Strength (1.00)",
    min = 0,
    max = 100,
    value = 100,
    onchange = function()
      dlg:modify{ id = "denoise",
        label = string.format("Strength (%.2f)", dlg.data.denoise / 100.0) }
    end,
  }

  dlg:slider{
    id = "steps",
    label = "Steps",
    min = 1,
    max = 100,
    value = 8,
  }

  dlg:slider{
    id = "cfg_scale",
    label = "CFG (5.0)",
    min = 0,
    max = 300,
    value = 50,
    onchange = function()
      dlg:modify{ id = "cfg_scale",
        label = string.format("CFG (%.1f)", dlg.data.cfg_scale / 10.0) }
    end,
  }

  dlg:slider{
    id = "clip_skip",
    label = "CLIP Skip",
    min = 1,
    max = 12,
    value = 2,
  }

  -- ══════════════════════════════════════════════════════════
  -- TAB: Post-Process
  -- ══════════════════════════════════════════════════════════
  dlg:tab{ id = "tab_pp", text = "Post-Process" }

  dlg:check{
    id = "pixelate",
    label = "Pixelate",
    selected = true,
  }

  dlg:slider{
    id = "pixel_size",
    label = "Target Size",
    min = 8,
    max = 512,
    value = 128,
  }

  dlg:slider{
    id = "colors",
    label = "Colors",
    min = 2,
    max = 256,
    value = 32,
  }

  dlg:combobox{
    id = "quantize_method",
    label = "Quantize",
    options = { "kmeans", "median_cut", "octree" },
    option = "kmeans",
  }

  dlg:combobox{
    id = "dither",
    label = "Dithering",
    options = { "none", "floyd_steinberg", "bayer_2x2", "bayer_4x4", "bayer_8x8" },
    option = "none",
  }

  dlg:combobox{
    id = "palette_mode",
    label = "Palette",
    options = { "auto", "preset", "custom" },
    option = "auto",
    onchange = function()
      local m = dlg.data.palette_mode
      dlg:modify{ id = "palette_name", visible = (m == "preset") }
      dlg:modify{ id = "palette_custom_colors", visible = (m == "custom") }
    end,
  }

  dlg:combobox{
    id = "palette_name",
    label = "Preset",
    options = { "pico8" },
    option = "pico8",
    visible = false,
  }

  dlg:entry{
    id = "palette_custom_colors",
    label = "Custom Hex",
    text = "",
    visible = false,
    hexpand = true,
  }

  dlg:check{
    id = "remove_bg",
    label = "Remove BG",
    selected = false,
  }

  -- ══════════════════════════════════════════════════════════
  -- TAB: Animation
  -- ══════════════════════════════════════════════════════════
  dlg:tab{ id = "tab_anim", text = "Animation" }

  dlg:combobox{
    id = "anim_method",
    label = "Method",
    options = { "chain", "animatediff" },
    option = "chain",
    onchange = function()
      local ad = dlg.data.anim_method == "animatediff"
      dlg:modify{ id = "anim_freeinit", visible = ad }
      dlg:modify{ id = "anim_freeinit_iters", visible = ad }
    end,
  }

  dlg:slider{
    id = "anim_frames",
    label = "Frames",
    min = 2,
    max = 120,
    value = 8,
  }

  dlg:slider{
    id = "anim_duration",
    label = "Duration (ms)",
    min = 50,
    max = 2000,
    value = 100,
  }

  dlg:slider{
    id = "anim_denoise",
    label = "Denoise (0.30)",
    min = 5,
    max = 100,
    value = 30,
    onchange = function()
      dlg:modify{ id = "anim_denoise",
        label = string.format("Denoise (%.2f)", dlg.data.anim_denoise / 100.0) }
    end,
  }

  dlg:combobox{
    id = "anim_seed_strategy",
    label = "Seed Mode",
    options = { "increment", "fixed", "random" },
    option = "increment",
  }

  dlg:entry{
    id = "anim_tag",
    label = "Tag Name",
    text = "",
    hexpand = true,
  }

  dlg:check{
    id = "anim_freeinit",
    label = "FreeInit",
    selected = false,
    visible = false,
  }

  dlg:slider{
    id = "anim_freeinit_iters",
    label = "FreeInit Iters",
    min = 1,
    max = 3,
    value = 2,
    visible = false,
  }

  -- ══════════════════════════════════════════════════════════
  -- TAB: Live
  -- ══════════════════════════════════════════════════════════
  dlg:tab{ id = "tab_live", text = "Live" }

  dlg:slider{
    id = "live_strength",
    label = "Strength (0.50)",
    min = 5,
    max = 95,
    value = 50,
    onchange = function()
      dlg:modify{ id = "live_strength",
        label = string.format("Strength (%.2f)", dlg.data.live_strength / 100.0) }
      if live_mode then
        send({ action = "realtime_update", denoise_strength = dlg.data.live_strength / 100.0 })
      end
    end,
  }

  dlg:slider{
    id = "live_steps",
    label = "Steps",
    min = 2,
    max = 8,
    value = 4,
    onchange = function()
      if live_mode then
        send({ action = "realtime_update", steps = dlg.data.live_steps })
      end
    end,
  }

  dlg:slider{
    id = "live_cfg",
    label = "CFG (2.5)",
    min = 10,
    max = 100,
    value = 25,
    onchange = function()
      dlg:modify{ id = "live_cfg",
        label = string.format("CFG (%.1f)", dlg.data.live_cfg / 10.0) }
      if live_mode then
        send({ action = "realtime_update", cfg_scale = dlg.data.live_cfg / 10.0 })
      end
    end,
  }

  dlg:slider{
    id = "live_opacity",
    label = "Preview (70%)",
    min = 10,
    max = 100,
    value = 70,
    onchange = function()
      dlg:modify{ id = "live_opacity",
        label = string.format("Preview (%d%%)", dlg.data.live_opacity) }
      if live_preview_layer then
        live_preview_layer.opacity = math.floor(dlg.data.live_opacity * 255 / 100)
        app.refresh()
      end
    end,
  }

  -- ── End tabs ──
  dlg:endtabs{ id = "main_tabs", selected = "tab_gen" }

  -- ══════════════════════════════════════════════════════════
  -- ACTIONS (always visible, bottom)
  -- ══════════════════════════════════════════════════════════
  dlg:separator{ text = "Actions" }

  dlg:button{
    id = "generate_btn",
    text = "GENERATE",
    enabled = false,
    hexpand = true,
    onclick = function()
      if generating or animating then return end
      local gw, gh = parse_size()
      local req = {
        action = "generate",
        prompt = dlg.data.prompt,
        negative_prompt = dlg.data.negative_prompt,
        mode = dlg.data.mode,
        width = gw, height = gh,
        seed = parse_seed(),
        steps = dlg.data.steps,
        cfg_scale = dlg.data.cfg_scale / 10.0,
        clip_skip = dlg.data.clip_skip,
        denoise_strength = dlg.data.denoise / 100.0,
        post_process = build_post_process(),
      }
      attach_lora(req)
      attach_neg_ti(req)
      if not attach_source_image(req) then return end

      generating = true
      gen_step_start = os.clock()
      dlg:modify{ id = "generate_btn", enabled = false }
      dlg:modify{ id = "cancel_btn", enabled = true }
      update_status("Generating...")
      send(req)
    end,
  }

  dlg:button{
    id = "cancel_btn",
    text = "CANCEL",
    enabled = false,
    onclick = function()
      if generating or animating then
        send({ action = "cancel" })
        update_status("Cancelling...")
      end
    end,
  }

  dlg:button{
    id = "animate_btn",
    text = "ANIMATE",
    enabled = false,
    hexpand = true,
    onclick = function()
      if animating or generating then return end
      local gw, gh = parse_size()
      local tag_name = dlg.data.anim_tag or ""
      if tag_name == "" then tag_name = nil end

      local req = {
        action = "generate_animation",
        method = dlg.data.anim_method,
        prompt = dlg.data.prompt,
        negative_prompt = dlg.data.negative_prompt,
        mode = dlg.data.mode,
        width = gw, height = gh,
        seed = parse_seed(),
        steps = dlg.data.steps,
        cfg_scale = dlg.data.cfg_scale / 10.0,
        clip_skip = dlg.data.clip_skip,
        denoise_strength = dlg.data.anim_denoise / 100.0,
        frame_count = dlg.data.anim_frames,
        frame_duration_ms = dlg.data.anim_duration,
        seed_strategy = dlg.data.anim_seed_strategy,
        tag_name = tag_name,
        enable_freeinit = dlg.data.anim_freeinit,
        freeinit_iterations = dlg.data.anim_freeinit_iters,
        post_process = build_post_process(),
      }
      attach_lora(req)
      attach_neg_ti(req)
      if not attach_source_image(req) then return end

      animating = true
      gen_step_start = os.clock()
      dlg:modify{ id = "animate_btn", enabled = false }
      dlg:modify{ id = "cancel_btn", enabled = true }
      update_status("Animating...")
      send(req)
    end,
  }

  dlg:button{
    id = "live_btn",
    text = "START LIVE",
    enabled = false,
    hexpand = true,
    onclick = function()
      if live_mode then
        send({ action = "realtime_stop" })
        stop_live_timer()
        update_status("Stopping live...")
      else
        if generating or animating then return end
        local spr = app.sprite
        if spr == nil then
          app.alert("Open a sprite first to use Live mode.")
          return
        end
        local gw, gh = parse_size()
        local req = {
          action = "realtime_start",
          prompt = dlg.data.prompt,
          negative_prompt = dlg.data.negative_prompt,
          width = gw, height = gh,
          seed = parse_seed(),
          steps = dlg.data.live_steps,
          cfg_scale = dlg.data.live_cfg / 10.0,
          denoise_strength = dlg.data.live_strength / 100.0,
          clip_skip = dlg.data.clip_skip,
          post_process = build_post_process(),
        }
        attach_lora(req)
        attach_neg_ti(req)
        live_canvas_hash = nil
        live_frame_id = 0
        update_status("Starting live...")
        send(req)
      end
    end,
  }

  dlg:button{
    id = "live_accept_btn",
    text = "ACCEPT",
    visible = false,
    onclick = function()
      local spr = app.sprite
      if spr == nil or live_preview_layer == nil then return end
      local cel = live_preview_layer:cel(app.frame)
      if cel == nil or cel.image == nil then return end
      local new_layer = spr:newLayer()
      new_layer.name = "PixyToon Live"
      spr:newCel(new_layer, app.frame, cel.image:clone(), cel.position)
      app.refresh()
      update_status("Live result accepted")
    end,
  }

  dlg:show{ wait = false, autoscrollbars = true }
end

-- ─── LAUNCH ──────────────────────────────────────────────────

build_dialog()
