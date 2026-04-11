from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

hiddenimports = collect_submodules('datasets')
datas = collect_data_files('datasets', include_py_files=True)
datas += copy_metadata('datasets')

module_collection_mode = {
    'datasets': 'pyz+py',
}