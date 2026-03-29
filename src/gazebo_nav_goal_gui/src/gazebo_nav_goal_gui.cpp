#include <gazebo/gui/gui.hh>
#include <gazebo/gui/GuiPlugin.hh>

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>

#include <sdf/sdf.hh>

#include <QWidget>
#include <QFrame>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QFont>
#include <QString>

#include <memory>
#include <string>

namespace gazebo
{

class GazeboNavGoalGui : public GUIPlugin
{
  Q_OBJECT

public:
  GazeboNavGoalGui()
      : GUIPlugin(),
        topic_("/move_base_simple/goal"),
        default_frame_("odom"),
        pos_x_(170),
        pos_y_(20)
  {
    this->setObjectName("gazeboNavGoalGui");
    this->resize(620, 430);
    this->move(pos_x_, pos_y_);

    auto *outerLayout = new QVBoxLayout();
    outerLayout->setContentsMargins(0, 0, 0, 0);
    outerLayout->setSpacing(0);

    auto *panel = new QFrame();
    panel->setObjectName("mainPanel");
    panel->setFixedSize(620, 430);

    panel->setStyleSheet(
        "#mainPanel {"
        "  background-color: rgba(24, 24, 24, 245);"
        "  border: 2px solid #3f3f3f;"
        "}"
        "QLabel {"
        "  color: white;"
        "}"
        "QLineEdit {"
        "  background: white;"
        "  color: black;"
        "  border: 1px solid #666;"
        "  border-radius: 2px;"
        "  padding-left: 8px;"
        "  padding-right: 8px;"
        "}"
        "QPushButton {"
        "  background-color: #5f5f5f;"
        "  color: white;"
        "  border: 1px solid #8a8a8a;"
        "  font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "  background-color: #737373;"
        "}"
    );

    auto *mainLayout = new QVBoxLayout(panel);
    mainLayout->setContentsMargins(18, 18, 18, 18);
    mainLayout->setSpacing(14);

    auto *title = new QLabel("2D Nav Goal");
    QFont titleFont("Sans", 20, QFont::Bold);
    title->setFont(titleFont);
    title->setFixedHeight(34);

    auto *desc = new QLabel(
        "Zadaj X, Y a yaw a plugin publikuje PoseStamped\n"
        "na /move_base_simple/goal pre move_base.");
    QFont descFont("Sans", 11);
    desc->setFont(descFont);
    desc->setWordWrap(true);
    desc->setFixedHeight(42);

    auto *grid = new QGridLayout();
    grid->setHorizontalSpacing(12);
    grid->setVerticalSpacing(10);

    QLabel *topicLabel = new QLabel("Topic:");
    QLabel *frameLabel = new QLabel("Frame:");
    QLabel *xLabel = new QLabel("X [m]:");
    QLabel *yLabel = new QLabel("Y [m]:");
    QLabel *yawLabel = new QLabel("Yaw [rad]:");

    QFont labelFont("Sans", 12, QFont::Bold);
    topicLabel->setFont(labelFont);
    frameLabel->setFont(labelFont);
    xLabel->setFont(labelFont);
    yLabel->setFont(labelFont);
    yawLabel->setFont(labelFont);

    topicLabel->setFixedHeight(28);
    frameLabel->setFixedHeight(28);
    xLabel->setFixedHeight(28);
    yLabel->setFixedHeight(28);
    yawLabel->setFixedHeight(28);

    topicEdit_ = new QLineEdit(QString::fromStdString(topic_));
    frameEdit_ = new QLineEdit(QString::fromStdString(default_frame_));
    xEdit_ = new QLineEdit("0.0");
    yEdit_ = new QLineEdit("0.0");
    yawEdit_ = new QLineEdit("0.0");

    QFont editFont("Sans", 12);
    topicEdit_->setFont(editFont);
    frameEdit_->setFont(editFont);
    xEdit_->setFont(editFont);
    yEdit_->setFont(editFont);
    yawEdit_->setFont(editFont);

    const int editHeight = 34;
    topicEdit_->setFixedHeight(editHeight);
    frameEdit_->setFixedHeight(editHeight);
    xEdit_->setFixedHeight(editHeight);
    yEdit_->setFixedHeight(editHeight);
    yawEdit_->setFixedHeight(editHeight);

    topicEdit_->setMinimumWidth(390);
    frameEdit_->setMinimumWidth(390);
    xEdit_->setMinimumWidth(390);
    yEdit_->setMinimumWidth(390);
    yawEdit_->setMinimumWidth(390);

    grid->addWidget(topicLabel, 0, 0);
    grid->addWidget(topicEdit_, 0, 1);
    grid->addWidget(frameLabel, 1, 0);
    grid->addWidget(frameEdit_, 1, 1);
    grid->addWidget(xLabel, 2, 0);
    grid->addWidget(xEdit_, 2, 1);
    grid->addWidget(yLabel, 3, 0);
    grid->addWidget(yEdit_, 3, 1);
    grid->addWidget(yawLabel, 4, 0);
    grid->addWidget(yawEdit_, 4, 1);

    grid->setColumnMinimumWidth(0, 95);
    grid->setColumnMinimumWidth(1, 390);
    grid->setColumnStretch(1, 1);

    for (int row = 0; row < 5; ++row)
      grid->setRowMinimumHeight(row, 36);

    auto *buttonRow = new QHBoxLayout();
    buttonRow->setSpacing(14);

    sendButton_ = new QPushButton("Send goal");
    zeroButton_ = new QPushButton("Zero");

    QFont buttonFont("Sans", 12, QFont::Bold);
    sendButton_->setFont(buttonFont);
    zeroButton_->setFont(buttonFont);

    sendButton_->setFixedHeight(46);
    zeroButton_->setFixedHeight(46);

    buttonRow->addWidget(sendButton_);
    buttonRow->addWidget(zeroButton_);

    statusLabel_ = new QLabel("Status: waiting");
    QFont statusFont("Sans", 11);
    statusLabel_->setFont(statusFont);
    statusLabel_->setWordWrap(true);
    statusLabel_->setFixedHeight(54);
    statusLabel_->setStyleSheet(
        "background-color: rgba(255,255,255,0.05);"
        "border: 1px solid #555;"
        "padding: 8px;"
        "color: #f0f0f0;");

    mainLayout->addWidget(title);
    mainLayout->addWidget(desc);
    mainLayout->addLayout(grid);
    mainLayout->addLayout(buttonRow);
    mainLayout->addWidget(statusLabel_);

    outerLayout->addWidget(panel);
    this->setLayout(outerLayout);

    connect(sendButton_, SIGNAL(clicked()), this, SLOT(OnSendGoal()));
    connect(zeroButton_, SIGNAL(clicked()), this, SLOT(OnZero()));

    InitializeRosPublisher();
  }

  virtual ~GazeboNavGoalGui()
  {
    if (ros::isStarted())
      ros_node_.reset();
  }

  void Load(sdf::ElementPtr _sdf) override
  {
    if (_sdf)
    {
      if (_sdf->HasElement("topic"))
        topic_ = _sdf->Get<std::string>("topic");

      if (_sdf->HasElement("default_frame"))
        default_frame_ = _sdf->Get<std::string>("default_frame");

      if (_sdf->HasElement("x"))
        pos_x_ = _sdf->Get<int>("x");

      if (_sdf->HasElement("y"))
        pos_y_ = _sdf->Get<int>("y");
    }

    this->move(pos_x_, pos_y_);
    topicEdit_->setText(QString::fromStdString(topic_));
    frameEdit_->setText(QString::fromStdString(default_frame_));

    InitializeRosPublisher();
    statusLabel_->setText(
        QString("Status: plugin loaded, topic=%1, frame=%2")
            .arg(QString::fromStdString(topic_))
            .arg(QString::fromStdString(default_frame_)));
  }

private slots:
  void OnZero()
  {
    xEdit_->setText("0.0");
    yEdit_->setText("0.0");
    yawEdit_->setText("0.0");
    statusLabel_->setText("Status: values reset to zero");
  }

  void OnSendGoal()
  {
    if (!ros::isInitialized())
    {
      statusLabel_->setText("Status: ROS nie je inicializovaný");
      return;
    }

    bool okX = false, okY = false, okYaw = false;

    const double x = xEdit_->text().toDouble(&okX);
    const double y = yEdit_->text().toDouble(&okY);
    const double yaw = yawEdit_->text().toDouble(&okYaw);

    if (!okX || !okY || !okYaw)
    {
      statusLabel_->setText("Status: neplatné čísla v X/Y/Yaw");
      return;
    }

    const std::string topic = topicEdit_->text().toStdString();
    const std::string frame = frameEdit_->text().toStdString();

    if (topic.empty())
    {
      statusLabel_->setText("Status: topic nesmie byť prázdny");
      return;
    }

    if (frame.empty())
    {
      statusLabel_->setText("Status: frame_id nesmie byť prázdny");
      return;
    }

    if (topic != topic_)
    {
      topic_ = topic;
      InitializeRosPublisher();
    }

    geometry_msgs::PoseStamped goal;
    goal.header.stamp = ros::Time::now();
    goal.header.frame_id = frame;
    goal.pose.position.x = x;
    goal.pose.position.y = y;
    goal.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);
    q.normalize();

    goal.pose.orientation.x = q.x();
    goal.pose.orientation.y = q.y();
    goal.pose.orientation.z = q.z();
    goal.pose.orientation.w = q.w();

    goal_pub_.publish(goal);

    statusLabel_->setText(
        QString("Status: sent -> frame=%1  x=%2  y=%3  yaw=%4")
            .arg(QString::fromStdString(frame))
            .arg(x, 0, 'f', 3)
            .arg(y, 0, 'f', 3)
            .arg(yaw, 0, 'f', 3));

    ROS_INFO_STREAM("[gazebo_nav_goal_gui] Published goal on " << topic_
                    << " frame=" << frame
                    << " x=" << x
                    << " y=" << y
                    << " yaw=" << yaw);
  }

private:
  void InitializeRosPublisher()
  {
    if (!ros::isInitialized())
      return;

    if (!ros_node_)
      ros_node_ = std::make_unique<ros::NodeHandle>("gazebo_nav_goal_gui");

    goal_pub_ = ros_node_->advertise<geometry_msgs::PoseStamped>(topic_, 1, true);
  }

private:
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::Publisher goal_pub_;

  std::string topic_;
  std::string default_frame_;
  int pos_x_;
  int pos_y_;

  QLineEdit *topicEdit_;
  QLineEdit *frameEdit_;
  QLineEdit *xEdit_;
  QLineEdit *yEdit_;
  QLineEdit *yawEdit_;
  QPushButton *sendButton_;
  QPushButton *zeroButton_;
  QLabel *statusLabel_;
};

GZ_REGISTER_GUI_PLUGIN(GazeboNavGoalGui)

}  // namespace gazebo

#include "gazebo_nav_goal_gui.moc"
