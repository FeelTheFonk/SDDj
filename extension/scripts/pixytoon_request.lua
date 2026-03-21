--
-- PixyToon — Request Builders
--

return function(PT)

function PT.parse_size()
  local s = PT.dlg.data.output_size
  local w, h = s:match("(%d+)x(%d+)")
  return tonumber(w) or 512, tonumber(h) or 512
end

function PT.parse_seed()
  local v = tonumber(PT.dlg.data.seed) or -1
  if v ~= math.floor(v) then v = -1 end
  return v
end

function PT.attach_lora(req)
  local sel = PT.dlg.data.lora_name
  if sel and sel ~= "(default)" then
    req.lora = { name = sel, weight = PT.dlg.data.lora_weight / 100.0 }
  end
end

function PT.attach_neg_ti(req)
  if PT.dlg.data.use_neg_ti and #PT.res.embeddings > 0 then
    local ti_list = {}
    local w = PT.dlg.data.neg_ti_weight / 100.0
    for _, name in ipairs(PT.res.embeddings) do
      ti_list[#ti_list + 1] = { name = name, weight = w }
    end
    req.negative_ti = ti_list
  end
end

function PT.attach_source_image(req)
  local mode = req.mode or "txt2img"
  if mode == "img2img" or mode:find("controlnet_") then
    local b64 = PT.capture_active_layer()
    if not b64 then
      app.alert("No active layer to use as source.")
      return false
    end
    if mode == "img2img" then req.source_image = b64
    else req.control_image = b64 end
  end
  if mode == "inpaint" then
    local src = PT.capture_flattened()
    if not src then
      app.alert("Inpaint requires an open sprite.")
      return false
    end
    req.source_image = src
    local mask = PT.capture_mask()
    if not mask then
      app.alert("Inpaint requires a mask.\n- Make a selection, or\n- Create a 'Mask' layer, or\n- Draw on active layer")
      return false
    end
    req.mask_image = mask
  end
  return true
end

function PT.build_post_process()
  local d = PT.dlg.data
  local pp = {
    pixelate = {
      enabled = d.pixelate,
      target_size = d.pixel_size,
    },
    quantize_method = d.quantize_method,
    quantize_colors = d.colors,
    dither = d.dither,
    palette = { mode = d.palette_mode },
    remove_bg = d.remove_bg,
  }
  if d.palette_mode == "preset" then
    pp.palette.name = d.palette_name
  elseif d.palette_mode == "custom" then
    local hex_str = d.palette_custom_colors or ""
    local colors = {}
    for hex in hex_str:gmatch("#?(%x%x%x%x%x%x)") do
      colors[#colors + 1] = "#" .. hex
    end
    if #colors > 0 then pp.palette.colors = colors end
  end
  return pp
end

-- Factored from generate button onclick + loop continuation (eliminates duplication).
function PT.build_generate_request()
  local gw, gh = PT.parse_size()
  local req = {
    action           = "generate",
    prompt           = PT.dlg.data.prompt,
    negative_prompt  = PT.dlg.data.negative_prompt,
    mode             = PT.dlg.data.mode,
    width            = gw,
    height           = gh,
    seed             = PT.parse_seed(),
    steps            = PT.dlg.data.steps,
    cfg_scale        = PT.dlg.data.cfg_scale / 10.0,
    clip_skip        = PT.dlg.data.clip_skip,
    denoise_strength = PT.dlg.data.denoise / 100.0,
    post_process     = PT.build_post_process(),
  }
  PT.attach_lora(req)
  PT.attach_neg_ti(req)
  return req
end

end
