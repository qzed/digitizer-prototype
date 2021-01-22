project(
    'test',
    'cpp',
    default_options: [
        'cpp_std=c++17',
        'optimization=3',
        'debug=false',
        'cpp_args=-ffast-math -mavx2',
        # 'cpp_args=-mavx2',
        'strip=true',
    ]
)

if get_option('access_checks')
    add_project_arguments('-DIPTSD_CONFIG_ACCESS_CHECKS', language: 'cpp')
endif

cairo_dep = dependency('cairo')

executable('main', 'src/main.cpp', dependencies: [cairo_dep])
