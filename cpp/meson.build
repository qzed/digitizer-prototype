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

cairo_dep = dependency('cairo')

executable('main', 'src/main.cpp', dependencies: [cairo_dep])
