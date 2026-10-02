#include "ggml-threading.h"
#include <mutex>

std::mutex ggml_critical_section_mutex;

void ggml_critical_section_start() {            // 加锁
    ggml_critical_section_mutex.lock();
}

void ggml_critical_section_end(void) {          // 解锁
    ggml_critical_section_mutex.unlock();
}
