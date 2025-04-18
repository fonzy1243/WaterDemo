#include "water_engine.h"

int main(int argc, char *argv[]) {
    WaterEngine engine;

    engine.init();

    engine.run();

    engine.cleanup();

    return 0;
}
