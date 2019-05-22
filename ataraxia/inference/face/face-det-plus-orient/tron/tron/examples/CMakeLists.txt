file(GLOB tmp test_*.cpp)
set(tron_examples_src ${tron_examples_src} ${tmp})

set(tron_examples_src ${tron_examples_src} PARENT_SCOPE)
