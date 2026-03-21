--
-- PixyToon — Constants & Shared State
--

return function(PT)

-- ─── Constants ──────────────────────────────────────────────

PT.cfg = {
  DEFAULT_SERVER_URL      = "ws://127.0.0.1:9876/ws",
  SETTINGS_FILE           = app.fs.joinPath(app.fs.userConfigPath, "pixytoon_settings.json"),
  CONNECT_TIMEOUT         = 5.0,
  HEARTBEAT_INTERVAL      = 30.0,
  GEN_TIMEOUT             = 300,
  LIVE_TIMER_INTERVAL     = 0.15,
  LIVE_COOLDOWN_INTERVAL  = 0.15,
  LIVE_INFLIGHT_TIMEOUT   = 10.0,
  LIVE_DEBOUNCE_INTERVAL  = 0.1,
  LOOP_DELAY              = 0.1,
  HASH_STEP_DIVISOR       = 16,
  DIRTY_STEP_DIVISOR      = 32,
}

-- ─── Mutable State ──────────────────────────────────────────

math.randomseed(os.time())

PT.ws_handle = nil
PT.dlg       = nil

PT.state = {
  connected      = false,
  generating     = false,
  animating      = false,
  cancel_pending = false,
  gen_step_start = nil,
  file_counter   = 0,
  session_id     = tostring(os.time()) .. "_" .. tostring(math.random(1000, 9999)),
}

PT.anim = {
  layer       = nil,
  start_frame = 0,
  frame_count = 0,
  base_seed   = 0,
}

PT.live = {
  mode             = false,
  timer            = nil,
  canvas_hash      = nil,
  frame_id         = 0,
  request_inflight = false,
  inflight_time    = nil,
  preview_layer    = nil,
  last_prompt      = nil,
  preview_sprite   = nil,
  prev_canvas      = nil,
  stroke_cooldown  = nil,
  cooldown_timer   = nil,
  cached_capture   = nil,
  slider_debounce  = nil,
}

PT.loop = {
  mode          = false,
  counter       = 0,
  seed_mode     = "random",
  random_mode   = false,
  locked_fields = {},
}

PT.res = {
  requested  = false,
  palettes   = {},
  loras      = {},
  embeddings = {},
  presets    = {},
}

PT.timers = {
  connect     = nil,
  heartbeat   = nil,
  gen_timeout = nil,
}

end
