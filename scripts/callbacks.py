from modules import devices, lowvram, script_callbacks, sd_hijack, shared
import scripts.text2vid as text2vid

script_callbacks.on_ui_tabs(text2vid.on_ui_tabs)
script_callbacks.on_ui_settings(text2vid.on_ui_settings)
