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
    , network(nullptr)
{
    ui->setupUi(this);

    trainButton = ui->trainButton;
    browseButton = ui->browseButton;
    testButton = ui->testButton;
    lineEdit = ui->lineEdit;
    resultLabel = ui->label;

    updateUI();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_trainButton_clicked()
{
    try {
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

        network = std::make_unique<NeuronNet>(std::list<NeuronNet::Pattern>(trainingPattern.begin(), trainingPattern.end()));
        QMessageBox::information(this, "Success", QString("Network trained with %1 patterns").arg(trainingPattern.size()));
    } catch (const std::exception &e) {
        QMessageBox::critical(this, "Error", QString("Training failed: ") + e.what());
    }

    updateUI();
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
    if (!network) {
        QMessageBox::warning(this, "Warning", "Please train the network first");
        return;
    }

    QString imagePath = lineEdit->text();

    if (imagePath.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please select a test file first");
        return;
    }

    try {
        auto pattern = ImageProcessor::loadImage(imagePath.toStdString());
        std::size_t steps = network->recognize(pattern);
        QFileInfo fileInfo(imagePath);
        QString resultPath = fileInfo.path() + "/recognized_" + fileInfo.fileName();
        ImageProcessor::saveImage(pattern, resultPath.toStdString(), imageWidth, imageHeight);
        showImage(pattern, imageWidth, imageHeight);
        resultLabel->setText(QString("Recognition completed in %1 steps. Result saved to %2").arg(steps).arg(resultPath));
    } catch (const std::exception &e) {
        QMessageBox::critical(this, "Error", QString("Recognition failed: ") + e.what());
    }
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

void MainWindow::updateUI() {
    bool isTrained = (network != nullptr);
    testButton->setEnabled(isTrained);
    resultLabel->setText(QString(isTrained ? "Network is ready for testing" : "Please train the network first"));
}
