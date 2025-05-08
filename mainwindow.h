#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QThread>

#include <vector>
#include <list>

#include "neural_worker.h"
#include "image_processor.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_trainButton_clicked();
    void on_browseButton_clicked();
    void on_testButton_clicked();

private:
    Ui::MainWindow *ui;
    QPushButton *trainButton;
    QPushButton *testButton;
    QPushButton *browseButton;
    QLineEdit *lineEdit;
    QLabel *resultLabel;

    QThread workerThread;
    NeuralWorker *worker;
    std::vector<NeuronNet::Pattern> trainingPattern;

    int imageWidth = 0;
    int imageHeight = 0;

    void updateUI(bool isTrained);
    void showImage(const std::vector<NeuronNet::State> &pattern, int width, int height);
};
#endif // MAINWINDOW_H
