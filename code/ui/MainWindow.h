#pragma once

#include <QMainWindow>

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

private:
    void appendLog(const QString& line);

    QPushButton* runButton_;
    QPushButton* clearButton_;
    QTextEdit* logView_;
};
