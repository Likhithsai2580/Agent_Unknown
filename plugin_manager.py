import importlib
import os
import time

class PluginManager:
    def __init__(self, plugin_dir):
        self.plugin_dir = plugin_dir
        self.plugins = {}
        self.last_loaded = {}

    def load_plugins(self):
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith('.py'):
                plugin_name = filename[:-3]
                file_path = os.path.join(self.plugin_dir, filename)
                if plugin_name not in self.plugins or os.path.getmtime(file_path) > self.last_loaded.get(plugin_name, 0):
                    spec = importlib.util.spec_from_file_location(plugin_name, file_path)
                    plugin_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(plugin_module)
                    if hasattr(plugin_module, 'register_plugin'):
                        self.plugins[plugin_name] = plugin_module.register_plugin()
                        self.last_loaded[plugin_name] = time.time()
                    print(f"Loaded plugin: {plugin_name}")

    def get_plugin(self, name):
        return self.plugins.get(name)
