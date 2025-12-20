#pragma once

#include <QMainWindow>
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>

class QPushButton;
class QTextEdit;
class QString;

class MainWindow : public QMainWindow {
	Q_OBJECT
public:
    	explicit MainWindow(QWidget* parent = nullptr);

private slots:
    	void runDemo();
    	void clearLog();
    	void onDemoFinished();
    	void appendLog(const QString& line);

private:
    	QPushButton* runButton_;
    	QPushButton* clearButton_;
    	QTextEdit* logView_;
    	QFutureWatcher<int> demoWatcher_;
};
