#include "MainWindow.h"

#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <QMetaObject>

#include "app/DemoRunner.h"

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent), runButton_(nullptr), clearButton_(nullptr), logView_(nullptr) {
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
    	connect(&demoWatcher_, &QFutureWatcher<int>::finished, this, &MainWindow::onDemoFinished);
}

void MainWindow::runDemo() {
    	runButton_->setEnabled(false);
    	appendLog("Starting demo...");

    	// The lambda that will be called from the background thread
    	auto logCallback = [this](const std::string& line) {
        	// Use invokeMethod to safely call appendLog on the GUI thread
        	QMetaObject::invokeMethod(this, "appendLog", Qt::QueuedConnection, Q_ARG(QString, QString::fromStdString(line)));
    	};

    	// Run the demo function in a separate thread
    	QFuture<int> future = QtConcurrent::run(&DemoRunner::runFullDemo, logCallback);
    	demoWatcher_.setFuture(future);
}

void MainWindow::onDemoFinished() {
	int result = demoWatcher_.result();
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
