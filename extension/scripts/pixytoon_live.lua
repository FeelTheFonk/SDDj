--
-- PixyToon — Live Paint System
--

return function(PT)

-- ─── Canvas Hashing ─────────────────────────────────────────

function PT.canvas_hash(img)
  local w, h = img.width, img.height
  local hash = 0
  local step = math.max(1, math.floor(math.min(w, h) / PT.cfg.HASH_STEP_DIVISOR))
  for y = 0, h - 1, step do
    for x = 0, w - 1, step do
      hash = (hash * 31 + img:getPixel(x, y)) % 2147483647
    end
  end
  return hash
end

-- ─── Dirty Region Detection ─────────────────────────────────

function PT.detect_dirty_region(prev, curr)
  local w, h = curr.width, curr.height
  local min_x, min_y = w, h
  local max_x, max_y = 0, 0
  local step = math.max(1, math.floor(math.min(w, h) / PT.cfg.DIRTY_STEP_DIVISOR))
  local found = false
  for y = 0, h - 1, step do
    for x = 0, w - 1, step do
      if prev:getPixel(x, y) ~= curr:getPixel(x, y) then
        found = true
        if x < min_x then min_x = x end
        if y < min_y then min_y = y end
        if x > max_x then max_x = x end
        if y > max_y then max_y = y end
      end
    end
  end
  if not found then return nil end
  min_x = math.max(0, min_x - step)
  min_y = math.max(0, min_y - step)
  max_x = math.min(w - 1, max_x + step)
  max_y = math.min(h - 1, max_y + step)
  return { x = min_x, y = min_y, w = max_x - min_x + 1, h = max_y - min_y + 1 }
end

-- ─── Timer Lifecycle ────────────────────────────────────────

function PT.stop_live_timer()
  PT.live.timer = PT.stop_timer(PT.live.timer)
  PT.live.request_inflight = false
  PT.live.inflight_time = nil
  PT.live.cooldown_timer = PT.stop_timer(PT.live.cooldown_timer)
  PT.live.cached_capture = nil
  PT.live.slider_debounce = PT.stop_timer(PT.live.slider_debounce)
end

-- Sends the cached capture as a realtime frame (called by cooldown debounce).
local function send_live_frame()
  if not PT.live.mode or PT.live.request_inflight then return end

  local curr_img = PT.live.cached_capture
  if not curr_img then return end

  -- ROI detection (bounding box only)
  local roi = nil
  if PT.live.prev_canvas
      and PT.live.prev_canvas.width == curr_img.width
      and PT.live.prev_canvas.height == curr_img.height then
    roi = PT.detect_dirty_region(PT.live.prev_canvas, curr_img)
  end
  PT.live.prev_canvas = curr_img:clone()

  local png_data = PT.image_to_png_bytes(curr_img)
  if not png_data then return end
  PT.live.frame_id = PT.live.frame_id + 1
  local header = {
    action = "realtime_frame",
    frame_id = PT.live.frame_id,
  }
  if roi then
    header.roi_x = roi.x
    header.roi_y = roi.y
    header.roi_w = roi.w
    header.roi_h = roi.h
  end
  local sent = PT.send_live_binary(header, png_data)
  if sent then
    PT.live.request_inflight = true
    PT.live.inflight_time = os.clock()
    if PT.dlg then PT.update_status("Live — processing...") end
  end
end

function PT.start_live_timer()
  PT.stop_live_timer()
  PT.live.prev_canvas = nil
  PT.live.stroke_cooldown = nil
  PT.live.cached_capture = nil

  -- Main polling timer
  PT.live.timer = Timer{
    interval = PT.cfg.LIVE_TIMER_INTERVAL,
    ontick = function()
      -- Inflight timeout guard: auto-reset after configured timeout
      if PT.live.request_inflight and PT.live.inflight_time then
        if (os.clock() - PT.live.inflight_time) > PT.cfg.LIVE_INFLIGHT_TIMEOUT then
          PT.live.request_inflight = false
          PT.live.inflight_time = nil
          if PT.dlg then PT.update_status("Live — timeout, retrying...") end
        end
      end

      -- Detect prompt changes
      if PT.live.mode and not PT.live.request_inflight and PT.dlg then
        local current_prompt = PT.dlg.data.prompt
        if current_prompt ~= PT.live.last_prompt then
          PT.live.last_prompt = current_prompt
          PT.send({ action = "realtime_update", prompt = current_prompt })
        end
      end

      if not PT.live.mode or not PT.state.connected or PT.live.request_inflight then return end
      local spr = app.sprite
      if spr == nil or app.frame == nil then
        pcall(function() PT.send({ action = "realtime_stop" }) end)
        PT.stop_live_timer()
        PT.live.mode = false
        PT.live.request_inflight = false
        PT.live.inflight_time = nil
        PT.live.preview_layer = nil
        PT.live.preview_sprite = nil
        if PT.dlg then
          PT.update_status("Live stopped (sprite closed)")
          PT.dlg:modify{ id = "live_btn", text = "START LIVE" }
          PT.dlg:modify{ id = "live_accept_btn", visible = false }
          PT.dlg:modify{ id = "generate_btn", enabled = true }
          PT.dlg:modify{ id = "animate_btn", enabled = true }
        end
        return
      end

      -- Single capture per cycle: hide preview layer for clean capture
      local was_visible = true
      if PT.live.preview_layer then
        local ok_vis, vis = pcall(function() return PT.live.preview_layer.isVisible end)
        if ok_vis then
          was_visible = vis
          PT.live.preview_layer.isVisible = false
        else
          PT.live.preview_layer = nil
        end
      end
      local flat_img = Image(spr.spec)
      flat_img:drawSprite(spr, app.frame)
      if PT.live.preview_layer then
        pcall(function() PT.live.preview_layer.isVisible = was_visible end)
      end

      -- Check if canvas changed
      local hash = PT.canvas_hash(flat_img)
      if hash == PT.live.canvas_hash then return end
      PT.live.canvas_hash = hash

      -- Cache capture for cooldown callback (avoids second drawSprite)
      PT.live.cached_capture = flat_img

      -- Debounce via cooldown: waits for drawing to pause before sending.
      -- If no frame is inflight, send immediately (first change after idle).
      PT.live.cooldown_timer = PT.stop_timer(PT.live.cooldown_timer)
      if not PT.live.stroke_cooldown then
        -- First change after idle: send immediately
        PT.live.stroke_cooldown = os.clock()
        send_live_frame()
      else
        -- Active drawing: debounce until pause
        PT.live.stroke_cooldown = os.clock()
        PT.live.cooldown_timer = Timer{
          interval = PT.cfg.LIVE_COOLDOWN_INTERVAL,
          ontick = function()
            PT.live.cooldown_timer = PT.stop_timer(PT.live.cooldown_timer)
            send_live_frame()
          end,
        }
        PT.live.cooldown_timer:start()
      end
    end,
  }
  PT.live.timer:start()
end

end
