from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# 收集 pandas._libs 所有子模組
hiddenimports = collect_submodules('pandas._libs')
hiddenimports += collect_submodules('pandas.core')
hiddenimports += collect_submodules('pandas.io')

# 收集 data files
datas = collect_data_files('pandas')
