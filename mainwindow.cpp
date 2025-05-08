#include <QMessageBox>
#include <QString>
#include <QDir>
#include <QStringList>
#include <QImage>
#include <QPixmap>
#include <QFileDialog>
#include <QFileInfo>

#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    trainButton = ui->trainButton;
    browseButton = ui->browseButton;
    testButton = ui->testButton;
    lineEdit = ui->lineEdit;
    resultLabel = ui->label;

    updateUI(false);

    worker = new NeuralWorker;
    worker->moveToThread(&workerThread);

    connect(worker, &NeuralWorker::trainingCompleted, this, [this]() -> void {
        QMessageBox::information(this, "success", "Training completed!");
        updateUI(true);
    });

    connect(worker, &NeuralWorker::recognitionCompleted, this, [this](NeuronNet::Pattern pattern, int steps) -> void {
        showImage(pattern, imageWidth, imageHeight);
        //resultLabel->setText(QString("Recognition completed in %1 steps").arg(steps));
    });

    connect(worker, &NeuralWorker::errorOccured, this, [this](QString msg) -> void {
        QMessageBox::critical(this, "Error", msg);
    });

    workerThread.start();
}

MainWindow::~MainWindow()
{
    workerThread.quit();
    workerThread.wait();
    delete ui;
}

void MainWindow::on_trainButton_clicked()
{
    trainingPattern.clear();

    QDir resourceDir("./resources");
    QStringList imageFiles = resourceDir.entryList({"*.png", "*.jpg"}, QDir::Files);

    if (imageFiles.empty()) {
        QMessageBox::warning(this, "Warning", "No training images found in resource directory");
        return;
    }

    for (const QString &file : imageFiles) {
        QString path = resourceDir.absoluteFilePath(file);
        std::vector<NeuronNet::State> pattern = ImageProcessor::loadImage(path.toStdString());

        if (imageWidth == 0) {
            QImage img(path);
            imageWidth = img.width();
            imageHeight = img.height();
        }

        trainingPattern.push_back(pattern);
    }

    Q_EMIT worker->trainNetwork(std::list<NeuronNet::Pattern>(trainingPattern.begin(), trainingPattern.end()));
}

void MainWindow::on_browseButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Select test image", "", "Image Files (*.png)");

    if (!fileName.isEmpty()) {
        lineEdit->setText(fileName);

        try {
            auto pattern = ImageProcessor::loadImage(fileName.toStdString());
            QImage img(fileName);
            showImage(pattern, img.width(), img.height());
        } catch (const std::exception &e) {
            QMessageBox::warning(this, "Error", QString("Failed to load image: ") + e.what());
        }
    }
}

void MainWindow::on_testButton_clicked()
{
    QString imagePath = lineEdit->text();

    if (imagePath.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please select a test file first");
        return;
    }

    auto pattern = ImageProcessor::loadImage(imagePath.toStdString());
    Q_EMIT worker->recognizePattern(pattern);
}

void MainWindow::showImage(const std::vector<NeuronNet::State> &pattern, int width, int height) {
    QImage img(width, height, QImage::Format_Grayscale8);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            img.setPixel(x, y, pattern[idx] == NeuronNet::State::Upper ? qRgb(0, 0, 0) : qRgb(255, 255, 255));
        }
    }

    QPixmap pixmap = QPixmap::fromImage(img);
    resultLabel->setPixmap(pixmap.scaled(resultLabel->size(), Qt::KeepAspectRatio));
}

void MainWindow::updateUI(bool isTrained) {
    testButton->setEnabled(isTrained);
    resultLabel->setText(QString(isTrained ? "Network is ready for testing" : "Please train the network first"));
}
