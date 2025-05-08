#ifndef NEURAL_WORKER_H
#define NEURAL_WORKER_H

#include <QObject>
#include <QString>

#include "neural_network.h"

class NeuralWorker : public QObject {
    Q_OBJECT

public:
    explicit NeuralWorker(QObject *parent = nullptr);

public slots:
    void trainNetwork(const std::list<NeuronNet::Pattern> &patterns);
    void recognizePattern(NeuronNet::Pattern pattern);

signals:
    void trainingCompleted();
    void recognitionCompleted(NeuronNet::Pattern pattern, int steps);
    void errorOccured(QString msg);

private:
    std::unique_ptr<NeuronNet> network;
};

#endif // NEURAL_WORKER_H
