#include "MainWindow.h"

#include <QApplication>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QWidget>

#include "../app/DemoRunner.h"

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), runButton_(nullptr), clearButton_(nullptr), logView_(nullptr) {
    auto* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);

    auto* title = new QLabel("Model Demo Runner", central);
    QFont titleFont = title->font();
    titleFont.setPointSize(14);
    titleFont.setBold(true);
    title->setFont(titleFont);

    runButton_ = new QPushButton("Run Demo", central);
    clearButton_ = new QPushButton("Clear Log", central);
    logView_ = new QTextEdit(central);
    logView_->setReadOnly(true);

    layout->addWidget(title);
    layout->addWidget(runButton_);
    layout->addWidget(clearButton_);
    layout->addWidget(logView_);
    central->setLayout(layout);
    setCentralWidget(central);

    setWindowTitle("MLS UI (Qt Widgets)");
    resize(720, 480);

    connect(runButton_, &QPushButton::clicked, this, &MainWindow::runDemo);
    connect(clearButton_, &QPushButton::clicked, this, &MainWindow::clearLog);
}

void MainWindow::runDemo() {
    runButton_->setEnabled(false);
    appendLog("Starting demo...");

    int result = DemoRunner::runFullDemo([this](const std::string& line) {
        appendLog(QString::fromStdString(line));
        QApplication::processEvents();
    });

    if (result == 0) {
        appendLog("Demo finished successfully.");
    } else {
        appendLog("Demo finished with errors.");
    }

    runButton_->setEnabled(true);
}

void MainWindow::clearLog() {
    logView_->clear();
}

void MainWindow::appendLog(const QString& line) {
    logView_->append(line);
}
