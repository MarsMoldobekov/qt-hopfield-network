#include "neural_worker.h"

NeuralWorker::NeuralWorker(QObject *parent) : QObject(parent) {
    network = std::make_unique<NeuronNet>(0);
}

void NeuralWorker::trainNetwork(const std::list<NeuronNet::Pattern> &patterns) {
    try {
        network->learn(patterns);
        emit trainingCompleted();
    } catch (const std::exception &e) {
        emit errorOccured(QString("Training failed:\n") + e.what());
    }
}

void NeuralWorker::recognizePattern(NeuronNet::Pattern pattern) {
    try {
        std::size_t steps = network->recognize(pattern);
        emit recognitionCompleted(pattern, steps);
    } catch (const std::exception &e) {
        emit errorOccured(QString("Recognition failed:\n") + e.what());
    }
}
