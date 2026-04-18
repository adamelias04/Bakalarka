#include <algorithm>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/Marker.h>

namespace
{

struct Point2D
{
  double x;
  double y;
};

// RobotMotionPredictor je tenky safety filter, ktory sedi medzi
// TEB local plannerom a robotom. Princip je jednoduchy:
//
//   1) Subscribe na /cmd_vel_raw (preremapovany TEB output) a na /odom
//      (aktualna poza robota).
//   2) Pre prijaty cmd_vel forward-simulujeme pohyb robota cez kratky
//      horizont (default 1.5 s) pomocou unicycle kinematiky:
//          x_{k+1} = x_k + v * cos(yaw_k) * dt
//          y_{k+1} = y_k + v * sin(yaw_k) * dt
//          yaw_{k+1} = yaw_k + w * dt
//      Ako (v, w) berieme commanded velocity z prichadzajuceho cmd_vel
//      (je to "intent" robota nasledujuci tick).
//   3) Kazdy bod predikcie skontrolujeme oproti lokalnej costmape. Cell
//      s hodnotou >= lethal_threshold (default 85, zhoda so
//      safe_zone_goal_router) povazujeme za potencialnu kolisiu.
//   4) Podla casu do prvej predikovanej kolizie (TTC) rozhodneme:
//          TTC >= slow_horizon  -> passthrough
//          stop < TTC < slow    -> linearne skalovat linear.x (a y)
//          rev  < TTC <= stop   -> stop (nechame angular.z)
//          TTC <= rev_horizon   -> reverse (ak povodne islo dopredu a
//                                            chrbat je volny), inak stop
//   5) Vystup publikujeme na /shoddy/cmd_vel (povodny cmd_vel topic
//      robota) a predikciu pre rviz na /robot_motion_prediction.
//
// Filter je zameny non-blocking: ak chyba odom alebo costmap, alebo ak
// je commanded velocity blizko nuly, prepustime cmd_vel bez zmien.
class RobotMotionPredictor
{
public:
  RobotMotionPredictor()
    : private_nh_("~"),
      has_odom_(false),
      has_costmap_(false)
  {
    private_nh_.param<std::string>("cmd_vel_in_topic", cmd_in_topic_, "/cmd_vel_raw");
    private_nh_.param<std::string>("cmd_vel_out_topic", cmd_out_topic_, "/shoddy/cmd_vel");
    private_nh_.param<std::string>("odom_topic", odom_topic_, "/odom");
    private_nh_.param<std::string>("local_costmap_topic", local_costmap_topic_,
                                   "/move_base/local_costmap/costmap");
    private_nh_.param<std::string>("obstacle_path_topic", obstacle_path_topic_,
                                   "/predicted_obstacle_path");
    private_nh_.param<std::string>("prediction_path_topic", prediction_path_topic_,
                                   "/robot_motion_prediction");
    private_nh_.param<std::string>("prediction_marker_topic", prediction_marker_topic_,
                                   "/robot_motion_prediction_marker");

    private_nh_.param("prediction_time", prediction_time_, 1.5);
    private_nh_.param("viz_rate", viz_rate_, 20.0);
    private_nh_.param("marker_line_width", marker_line_width_, 0.05);
    private_nh_.param("marker_z", marker_z_, 0.10);
    private_nh_.param("prediction_dt", prediction_dt_, 0.1);
    private_nh_.param("slow_horizon", slow_horizon_, 1.0);
    private_nh_.param("stop_horizon", stop_horizon_, 0.4);
    private_nh_.param("reverse_horizon", reverse_horizon_, 0.2);
    private_nh_.param("min_check_linear_velocity", min_check_linear_velocity_, 0.02);
    private_nh_.param("min_check_angular_velocity", min_check_angular_velocity_, 0.05);
    private_nh_.param("lethal_threshold", lethal_threshold_, 85);
    private_nh_.param("reverse_velocity", reverse_velocity_, -0.15);
    private_nh_.param("reverse_clear_probe_distance", reverse_clear_probe_distance_, 0.45);
    private_nh_.param("publish_prediction_path", publish_prediction_path_, true);

    // Temporal overlap check (porovnava robotovu predikciu s
    // /predicted_obstacle_path z dynamic_obstacle_predictor v rovnakych
    // casovych krokoch).
    private_nh_.param("temporal_safety_distance", temporal_safety_distance_, 0.48);
    private_nh_.param("temporal_path_max_age",    temporal_path_max_age_,    1.0);
    private_nh_.param("obstacle_prediction_dt",   obstacle_prediction_dt_,   0.1);
    private_nh_.param("temporal_prediction_time", temporal_prediction_time_, 3.0);

    // Escape mode: ak je robot pozicia v lethal bunke, vyhodnotime niekolko
    // kandidatov (v, w) a vyberieme ten, ktory najrychlejsie opusta lethal
    // zonu a drzi bezpecnu vzdialenost od obstacle prediction.
    private_nh_.param("escape_horizon",          escape_horizon_,          1.5);
    private_nh_.param("escape_linear_velocity",  escape_v_,                0.25);
    private_nh_.param("escape_angular_velocity", escape_w_,                0.6);

    // Head-on maneuver: ak temporalna klasifikacia hovori HEAD-ON, namiesto
    // passthrough porovname swerve-L/R + reverse a vyberieme kandidata,
    // ktory MAXIMALIZUJE min_obs_dist cez horizon. Strana-lock hysteréza
    // zabranuje ticavosti medzi L a R.
    private_nh_.param("headon_forward_velocity", headon_v_,                0.25);
    private_nh_.param("headon_swerve_angular",   headon_w_,                0.7);
    private_nh_.param("headon_reverse_velocity", headon_rev_v_,           -0.25);
    private_nh_.param("headon_horizon",          headon_horizon_,          2.0);
    private_nh_.param("headon_side_lock_time",   headon_side_lock_time_,   2.0);
    private_nh_.param("headon_side_lock_bonus",  headon_side_lock_bonus_,  0.25);
    private_nh_.param("headon_hard_lock",        headon_hard_lock_,        true);
    private_nh_.param("headon_min_feasible_steps", headon_min_feasible_steps_, 2);

    // Command latch: ked HEADON alebo LETHAL_ESCAPE vyberie reverse
    // maneuver, na dalsie ticky zamkneme ten cmd namiesto toho aby TEB
    // (ktory posiela ~10Hz forward max) okamzite prepisal reverze. Inak
    // robot necuva — zasiahne nas filter len raz za ~40 tickov a nestihne
    // fyzicky zreverovat. Latch expires po cmd_latch_time_ alebo ked
    // robot uz nie je v konflikte / lethal.
    private_nh_.param("cmd_latch_time",            cmd_latch_time_,           0.6);

    if (prediction_dt_ < 1e-3)
      prediction_dt_ = 1e-3;
    if (prediction_time_ < prediction_dt_)
      prediction_time_ = prediction_dt_;

    cmd_pub_ = nh_.advertise<geometry_msgs::Twist>(cmd_out_topic_, 1);
    path_pub_ = nh_.advertise<nav_msgs::Path>(prediction_path_topic_, 1);
    marker_pub_ = nh_.advertise<visualization_msgs::Marker>(prediction_marker_topic_, 1);

    cmd_sub_ = nh_.subscribe(cmd_in_topic_, 10, &RobotMotionPredictor::cmdCallback, this);
    odom_sub_ = nh_.subscribe(odom_topic_, 10, &RobotMotionPredictor::odomCallback, this);
    costmap_sub_ = nh_.subscribe(local_costmap_topic_, 1,
                                 &RobotMotionPredictor::costmapCallback, this);
    obstacle_path_sub_ = nh_.subscribe(obstacle_path_topic_, 1,
                                       &RobotMotionPredictor::obstaclePathCallback, this);

    if (viz_rate_ > 0.0)
    {
      viz_timer_ = nh_.createTimer(ros::Duration(1.0 / viz_rate_),
                                   &RobotMotionPredictor::vizTimerCallback, this);
    }

    ROS_INFO("RobotMotionPredictor: in=%s out=%s odom=%s costmap=%s horizon=%.2fs viz=%.1fHz",
             cmd_in_topic_.c_str(), cmd_out_topic_.c_str(),
             odom_topic_.c_str(), local_costmap_topic_.c_str(),
             prediction_time_, viz_rate_);
    ROS_INFO("RobotMotionPredictor: RESOLVED subscriptions: cmd='%s' odom='%s' costmap='%s' obstacle='%s'",
             cmd_sub_.getTopic().c_str(),
             odom_sub_.getTopic().c_str(),
             costmap_sub_.getTopic().c_str(),
             obstacle_path_sub_.getTopic().c_str());
  }

private:
  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_odom_ = *msg;
    has_odom_ = true;
  }

  void costmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_costmap_ = *msg;
    has_costmap_ = true;
  }

  void obstaclePathCallback(const nav_msgs::Path::ConstPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_obstacle_path_ = *msg;
    // dynamic_obstacle_predictor publikuje prazdnu Path pri vyprazdneni
    // tracku (clearPublishedPrediction), takze !poses.empty() je
    // spravny "stale-vs-cleared" guard.
    has_obstacle_path_ = !msg->poses.empty();
  }

  struct PredictionResult
  {
    nav_msgs::Path path;
    double t_collision;  // -1 ak ziadna kolizia v horizonte
  };

  // Escape: kandidat manevru + skorovaci vysledok.
  struct EscapeCandidate
  {
    double v;
    double w;
    const char* name;
  };

  struct EscapeScore
  {
    bool feasible = false;           // existuje krok, kde sme mimo lethal
    double t_exit = 0.0;             // sekundy do vystupu z lethal zony
    double min_obs_dist = std::numeric_limits<double>::infinity();
  };

  // Head-on: kandidat (rovnaka semantika ako Escape) + skorovaci vysledok.
  // Feasible znamena ze dratovany manever neprejde cez lethal bunku
  // pocas headon_horizon_. Vyberame max(min_obs_dist).
  struct HeadOnCandidate
  {
    double v;
    double w;
    int side;          // -1 = R, 0 = neutral, +1 = L (pre hysteresis)
    const char* name;
  };

  struct HeadOnScore
  {
    bool feasible = false;
    double min_obs_dist = std::numeric_limits<double>::infinity();
  };

  PredictionResult runPrediction(const nav_msgs::Odometry& odom,
                                  const nav_msgs::OccupancyGrid& costmap,
                                  double v, double w) const
  {
    PredictionResult res;
    res.t_collision = -1.0;
    res.path.header.stamp = ros::Time::now();
    res.path.header.frame_id = costmap.header.frame_id.empty()
                                   ? odom.header.frame_id
                                   : costmap.header.frame_id;

    const double yaw0 = yawFromQuaternion(odom.pose.pose.orientation);
    double cx = odom.pose.pose.position.x;
    double cy = odom.pose.pose.position.y;
    double cyaw = yaw0;

    const int n_steps = static_cast<int>(std::ceil(prediction_time_ / prediction_dt_));
    res.path.poses.reserve(n_steps);
    for (int i = 1; i <= n_steps; ++i)
    {
      cyaw += w * prediction_dt_;
      cx += v * std::cos(cyaw) * prediction_dt_;
      cy += v * std::sin(cyaw) * prediction_dt_;

      geometry_msgs::PoseStamped p;
      p.header = res.path.header;
      p.pose.position.x = cx;
      p.pose.position.y = cy;
      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, cyaw);
      p.pose.orientation = tf2::toMsg(q);
      res.path.poses.push_back(p);

      if (res.t_collision < 0.0 && cellLethal(costmap, cx, cy))
        res.t_collision = static_cast<double>(i) * prediction_dt_;
    }
    return res;
  }

  // Iba na ucely vizualizacie — bez REVERSE/HARDSTOP rozlisenia.
  const char* simpleMode(double t_collision) const
  {
    if (t_collision < 0.0 || t_collision >= slow_horizon_)
      return "PASS";
    if (t_collision >= stop_horizon_)
      return "SLOW";
    if (t_collision >= reverse_horizon_)
      return "STOP";
    return "HARDSTOP";
  }

  void cmdCallback(const geometry_msgs::Twist::ConstPtr& msg)
  {
    geometry_msgs::Twist out = *msg;

    nav_msgs::Odometry odom;
    nav_msgs::OccupancyGrid costmap;
    bool have_state = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (has_odom_ && has_costmap_)
      {
        odom = latest_odom_;
        costmap = latest_costmap_;
        have_state = true;
      }
    }

    // Bez state proste prepustime — nemozeme spravne rozhodnut.
    if (!have_state)
    {
      cmd_pub_.publish(out);
      return;
    }

    // ---- Command latch --------------------------------------------------
    //
    // Ak predchadzajuci tick zamol reverse maneuver (HEADON / ESCAPE),
    // drzime ten cmd kym latch neexpiruje. Bez toho TEB (~10Hz, forward=1.0)
    // kazdy druhy tick prepise reverze a robot sa fyzicky nepohne dozadu.
    {
      const ros::Time now_latch = ros::Time::now();
      if (!cmd_latch_until_.isZero() && now_latch < cmd_latch_until_)
      {
        ROS_WARN_THROTTLE(1.0,
                          "RobotMotionPredictor: %s LATCH hold (v=%.2f w=%.2f) "
                          "remaining=%.2fs cmd=(v=%.2f w=%.2f)",
                          cmd_latch_reason_.c_str(),
                          cmd_latch_value_.linear.x, cmd_latch_value_.angular.z,
                          (cmd_latch_until_ - now_latch).toSec(),
                          msg->linear.x, msg->angular.z);
        cmd_pub_.publish(cmd_latch_value_);
        return;
      }
    }

    // ---- Lethal escape mode ---------------------------------------------
    //
    // Ak je robot pozicia v lethal bunke, TEB vystup nestaci — aktivne
    // vyberieme escape manever (dopredu / dozadu / rotacia / kombinacia)
    // podla toho, ktory najrychlejsie unika a ktora trajektoria sa
    // najmenej blizi k predikcii prekazky.
    if (cellLethal(costmap, odom.pose.pose.position.x, odom.pose.pose.position.y))
    {
      nav_msgs::Path obs_path_local;
      const nav_msgs::Path* obs_ptr = nullptr;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (has_obstacle_path_)
        {
          obs_path_local = latest_obstacle_path_;
          obs_ptr = &obs_path_local;
        }
      }

      geometry_msgs::Twist esc;
      const char* esc_mode = nullptr;
      std::string esc_scores;
      if (selectEscape(odom, costmap, obs_ptr, esc, esc_mode, esc_scores))
      {
        ROS_WARN_THROTTLE(1.0,
                          "RobotMotionPredictor: LETHAL_ESCAPE picked=%s (v=%.2f w=%.2f) "
                          "(cmd v=%.2f w=%.2f) scores:%s",
                          esc_mode, esc.linear.x, esc.angular.z,
                          msg->linear.x, msg->angular.z,
                          esc_scores.c_str());
        // Zamkneme escape cmd aby TEB nestihol cez nasledujuce ticky
        // prepisat forward — inak robot nestihne fyzicky reverovat.
        if (esc.linear.x < 0.0)
        {
          cmd_latch_until_ = ros::Time::now() + ros::Duration(cmd_latch_time_);
          cmd_latch_value_ = esc;
          cmd_latch_reason_ = std::string("LETHAL_ESCAPE/") + esc_mode;
        }
        cmd_pub_.publish(esc);
        return;
      }

      ROS_WARN_THROTTLE(1.0,
                        "RobotMotionPredictor: LETHAL_NO_ESCAPE (hardstop, "
                        "cmd v=%.2f w=%.2f) scores:%s",
                        msg->linear.x, msg->angular.z,
                        esc_scores.c_str());
      out.linear.x = 0.0;
      out.linear.y = 0.0;
      out.angular.z = 0.0;
      cmd_pub_.publish(out);
      return;
    }

    const double v_cmd = msg->linear.x;
    const double w_cmd = msg->angular.z;

    // Mikropohyby ignorujeme, prepustime.
    if (std::fabs(v_cmd) < min_check_linear_velocity_ &&
        std::fabs(w_cmd) < min_check_angular_velocity_)
    {
      cmd_pub_.publish(out);
      return;
    }

    const PredictionResult pred = runPrediction(odom, costmap, v_cmd, w_cmd);
    const double t_collision = pred.t_collision;

    // Decide.
    const char* mode = "PASS";
    if (t_collision < 0.0 || t_collision >= slow_horizon_)
    {
      // Passthrough, predikcia je cista.
    }
    else if (t_collision >= stop_horizon_)
    {
      // Linear braking medzi slow_horizon a stop_horizon.
      const double span = std::max(slow_horizon_ - stop_horizon_, 1e-3);
      double scale = (t_collision - stop_horizon_) / span;
      scale = std::max(0.0, std::min(1.0, scale));
      out.linear.x = msg->linear.x * scale;
      out.linear.y = msg->linear.y * scale;
      // angular nechame, nech moze rotovat preč od prekazky
      mode = "SLOW";
    }
    else if (t_collision >= reverse_horizon_)
    {
      // Stop. Angular nechame, nech sa moze otacat.
      out.linear.x = 0.0;
      out.linear.y = 0.0;
      mode = "STOP";
    }
    else
    {
      // Imminent collision. Ak povodne sli dopredu a za nami je volno -> reverse.
      // Inak hard stop.
      const Point2D p0{odom.pose.pose.position.x, odom.pose.pose.position.y};
      const double yaw0 = yawFromQuaternion(odom.pose.pose.orientation);
      const bool was_forward = msg->linear.x > 0.0;
      const bool rear_clear = was_forward ? isRearClear(costmap, p0, yaw0) : false;
      if (was_forward && rear_clear)
      {
        out.linear.x = reverse_velocity_;
        out.linear.y = 0.0;
        out.angular.z = 0.0;
        mode = "REVERSE";
      }
      else
      {
        out.linear.x = 0.0;
        out.linear.y = 0.0;
        out.angular.z = 0.0;
        mode = "HARDSTOP";
      }
    }

    if (t_collision >= 0.0)
    {
      ROS_WARN_THROTTLE(1.0,
                        "RobotMotionPredictor: %s (TTC=%.2fs cmd v=%.2f w=%.2f -> v=%.2f w=%.2f)",
                        mode, t_collision, v_cmd, w_cmd, out.linear.x, out.angular.z);
    }

    // ---- Temporal overlap layer ---------------------------------------
    //
    // Porovnava robotovu predikciu s predikciou prekazky v rovnakych
    // casovych krokoch. Klasifikuje konflikt:
    //
    //   CROSSING  — zastavenie riesi konflikt (prekazka pretina cestu)
    //               -> TEMPORAL_STOP (v=0, angular zachovane pre TEB)
    //   HEAD-ON   — zastavenie neriesi (prekazka ide na nas)
    //               -> prepustime TEB cmd_vel — PredictedSweepRiskLayer
    //                  + costmap costs + via-points nech TEB uhne do strany,
    //                  staticky safety layer (SLOW/STOP/REVERSE) je poistka.
    nav_msgs::Path obs_path_local;
    bool have_obs_path_local = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (has_obstacle_path_)
      {
        obs_path_local = latest_obstacle_path_;
        have_obs_path_local = true;
      }
    }

    if (have_obs_path_local)
    {
      const TemporalConflict tc =
          checkTemporalConflict(odom, obs_path_local, out.linear.x, out.angular.z);

      // FOLLOWING skip: ak je prekazka pred robotom a ide priblizne rovnakym
      // smerom ako robot (dot(obstacle_vel_dir, robot_heading) > 0), nejde o
      // head-on ani crossing, ale o dobiehanie / predbiehanie. Temporal layer
      // v takom pripade zbytocne robota stopuje (TEMPORAL_STOP) a blokuje
      // TEB-u moznost obist cez via-points + costmap. Necame to na TEB.
      const bool following = isFollowingScenario(odom, obs_path_local);

      if (tc.present && !following)
      {
        const char* tmode = nullptr;

        // Klasifikacia: zastavenie riesi konflikt?
        const TemporalConflict tc_stop =
            checkTemporalConflict(odom, obs_path_local, 0.0, 0.0);

        if (!tc_stop.present)
        {
          // CROSSING — prekazka pretina cestu, zastavenie staci.
          // Angular zachovame, nech TEB moze rotovat.
          out.linear.x = 0.0;
          out.linear.y = 0.0;
          tmode = "TEMPORAL_STOP";

          ROS_WARN_THROTTLE(1.0,
                            "RobotMotionPredictor: %s (t_conflict=%.2fs dist=%.2fm "
                            "cmd v=%.2f w=%.2f -> v=%.2f w=%.2f)",
                            tmode, tc.t_conflict, tc.dist, v_cmd, w_cmd,
                            out.linear.x, out.angular.z);
        }
        else
        {
          // HEAD-ON — prekazka ide priamo na nas. Zastavenie neriesi,
          // pasivny passthrough na TEB tiez nezvlada (closing rate je
          // privysoka). Aktivne vyberieme swerve-L/R alebo reverse.
          geometry_msgs::Twist hd;
          const char* hd_mode = nullptr;
          std::string hd_scores;
          if (selectHeadOnManeuver(odom, costmap, obs_path_local,
                                    hd, hd_mode, hd_scores))
          {
            out.linear.x = hd.linear.x;
            out.linear.y = 0.0;
            out.angular.z = hd.angular.z;
            tmode = hd_mode;
            // Zamkneme HEADON REV cmd — bez toho TEB (~10Hz forward=1.0)
            // prepise reverze hned v dalsom ticku a robot nestihne
            // zreverovat ani ~10cm. Latch drzi reverse cmd po cmd_latch_time_.
            if (hd.linear.x < 0.0)
            {
              cmd_latch_until_ = ros::Time::now() + ros::Duration(cmd_latch_time_);
              cmd_latch_value_ = hd;
              cmd_latch_reason_ = std::string("HEADON/") + hd_mode;
            }
          }
          else
          {
            // Ziadny kandidat nie je feasible (vsetky vedu do lethal) —
            // hardstop, nech sa kocka prejde okolo nas.
            out.linear.x = 0.0;
            out.linear.y = 0.0;
            out.angular.z = 0.0;
            tmode = "TEMPORAL_HEADON_STUCK";
          }

          ROS_WARN_THROTTLE(1.0,
                            "RobotMotionPredictor: %s (t_conflict=%.2fs dist=%.2fm "
                            "cmd v=%.2f w=%.2f -> v=%.2f w=%.2f) scores:%s",
                            tmode, tc.t_conflict, tc.dist, v_cmd, w_cmd,
                            out.linear.x, out.angular.z,
                            hd_scores.c_str());
        }
      }
    }

    cmd_pub_.publish(out);
  }

  // Periodicky publikuje vizualizaciu predikcie podla AKTUALNEJ rychlosti
  // robota (z odom.twist). Beh nezavisi na cmd_vel — funguje aj pri
  // manualnom riadeni cez rqt_robot_steering.
  void vizTimerCallback(const ros::TimerEvent&)
  {
    if (!publish_prediction_path_)
      return;

    nav_msgs::Odometry odom;
    nav_msgs::OccupancyGrid costmap;
    bool have_odom_local = false;
    bool have_costmap_local = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      have_odom_local = has_odom_;
      have_costmap_local = has_costmap_;
      if (have_odom_local)
        odom = latest_odom_;
      if (have_costmap_local)
        costmap = latest_costmap_;
    }

    if (!have_odom_local || !have_costmap_local)
    {
      ROS_WARN_THROTTLE(2.0,
                        "RobotMotionPredictor viz: cakam na topicy (odom=%s costmap=%s). "
                        "Skontroluj `rostopic hz %s` a `rostopic hz %s`.",
                        have_odom_local ? "OK" : "MISSING",
                        have_costmap_local ? "OK" : "MISSING",
                        odom_topic_.c_str(), local_costmap_topic_.c_str());
      return;
    }

    const double v = odom.twist.twist.linear.x;
    const double w = odom.twist.twist.angular.z;

    const std::string frame = costmap.header.frame_id.empty()
                                  ? odom.header.frame_id
                                  : costmap.header.frame_id;

    // Stojaci robot — ziadna predikcia, marker zhasneme.
    if (std::fabs(v) < min_check_linear_velocity_ &&
        std::fabs(w) < min_check_angular_velocity_)
    {
      ROS_INFO_THROTTLE(5.0,
                        "RobotMotionPredictor viz: robot stoji (v=%.3f w=%.3f), marker zhasnuty.",
                        v, w);
      publishEmptyPath(frame);
      publishEmptyMarker(frame);
      return;
    }

    const PredictionResult pred = runPrediction(odom, costmap, v, w);

    // Temporal overlap check pre vizualizaciu — beha aj ked cmd_vel je
    // nezasiahnuty, aby uzivatel videl, ze filter bezi.
    nav_msgs::Path obs_path_local;
    bool have_obs_path_local = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (has_obstacle_path_)
      {
        obs_path_local = latest_obstacle_path_;
        have_obs_path_local = true;
      }
    }

    const char* mode = simpleMode(pred.t_collision);
    double t_conflict_viz = -1.0;
    if (have_obs_path_local)
    {
      const TemporalConflict tc = checkTemporalConflict(odom, obs_path_local, v, w);
      if (tc.present)
      {
        // Temporal ma prednost vo viz farbe (cyan), aby bolo jasne, ze
        // ide o dynamic obstacle a nie statickym costmap hit.
        mode = "TEMPORAL";
        t_conflict_viz = tc.t_conflict;
      }
    }

    path_pub_.publish(pred.path);
    publishMarker(pred.path, mode);

    if (t_conflict_viz >= 0.0)
    {
      ROS_INFO_THROTTLE(2.0,
                        "RobotMotionPredictor viz: publikujem %zu bodov (v=%.2f w=%.2f frame=%s mode=%s t_conflict=%.2fs)",
                        pred.path.poses.size(), v, w, frame.c_str(), mode, t_conflict_viz);
    }
    else
    {
      ROS_INFO_THROTTLE(2.0,
                        "RobotMotionPredictor viz: publikujem %zu bodov (v=%.2f w=%.2f frame=%s mode=%s)",
                        pred.path.poses.size(), v, w, frame.c_str(), mode);
    }
  }

  void publishMarker(const nav_msgs::Path& path, const char* mode)
  {
    visualization_msgs::Marker marker;
    marker.header = path.header;
    marker.ns = "robot_motion_prediction";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = marker_line_width_;
    marker.lifetime = ros::Duration(0.5);

    // Farba podla bezpecnostneho modu.
    //   PASS     -> zelena
    //   SLOW     -> zlta
    //   STOP     -> oranzova
    //   REVERSE  -> magenta
    //   HARDSTOP -> cervena
    //   TEMPORAL -> cyan (dynamic obstacle prediction overlap)
    std::string m = mode ? mode : "PASS";
    if (m == "SLOW")
    {
      marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0;
    }
    else if (m == "STOP")
    {
      marker.color.r = 1.0; marker.color.g = 0.5; marker.color.b = 0.0;
    }
    else if (m == "REVERSE")
    {
      marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 1.0;
    }
    else if (m == "HARDSTOP")
    {
      marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0;
    }
    else if (m == "TEMPORAL")
    {
      marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 1.0;
    }
    else
    {
      marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0;
    }
    marker.color.a = 0.9;

    marker.points.reserve(path.poses.size());
    for (const auto& p : path.poses)
    {
      geometry_msgs::Point pt;
      pt.x = p.pose.position.x;
      pt.y = p.pose.position.y;
      pt.z = marker_z_;
      marker.points.push_back(pt);
    }

    marker_pub_.publish(marker);
  }

  void publishEmptyMarker(const std::string& frame_id)
  {
    if (!publish_prediction_path_)
      return;
    visualization_msgs::Marker marker;
    marker.header.stamp = ros::Time::now();
    marker.header.frame_id = frame_id;
    marker.ns = "robot_motion_prediction";
    marker.id = 0;
    marker.action = visualization_msgs::Marker::DELETE;
    marker_pub_.publish(marker);
  }

  bool cellLethal(const nav_msgs::OccupancyGrid& cm, double wx, double wy) const
  {
    const double res = cm.info.resolution;
    if (res <= 0.0)
      return false;
    const double ox = cm.info.origin.position.x;
    const double oy = cm.info.origin.position.y;
    const int W = static_cast<int>(cm.info.width);
    const int H = static_cast<int>(cm.info.height);

    const int mx = static_cast<int>(std::floor((wx - ox) / res));
    const int my = static_cast<int>(std::floor((wy - oy) / res));
    if (mx < 0 || my < 0 || mx >= W || my >= H)
      return false;

    const std::size_t idx = static_cast<std::size_t>(my) * static_cast<std::size_t>(W) +
                            static_cast<std::size_t>(mx);
    if (idx >= cm.data.size())
      return false;
    const int v = static_cast<int>(cm.data[idx]);
    return v >= lethal_threshold_;
  }

  // Detekcia FOLLOW scenara: prekazka je vpredu v robotovom frame a ide
  // priblizne rovnakym smerom ako robot hlavi. V takom pripade NIE JE
  // head-on ani crossing — robot dobieha / ma prekazku obist. Temporal
  // layer sa tu nesmie do toho montat, inak by TEMPORAL_STOP zastavil
  // robota a TEB by nedokazal cez via-points ani obchadzat.
  bool isFollowingScenario(const nav_msgs::Odometry& odom,
                            const nav_msgs::Path& obs_path) const
  {
    if (obs_path.poses.size() < 2) return false;

    const ros::Time now = ros::Time::now();
    const double age = (now - obs_path.header.stamp).toSec();
    if (age > temporal_path_max_age_) return false;

    const auto& p0 = obs_path.poses.front().pose.position;
    const auto& p1 = obs_path.poses[1].pose.position;
    const double ovx = p1.x - p0.x;
    const double ovy = p1.y - p0.y;
    const double ospeed = std::hypot(ovx, ovy);
    if (ospeed < 1e-3) return false;  // prekazka stoji -> nie follow

    const double yaw = yawFromQuaternion(odom.pose.pose.orientation);
    const double hx = std::cos(yaw);
    const double hy = std::sin(yaw);

    // 1) prekazka je pred robotom
    const double rx = p0.x - odom.pose.pose.position.x;
    const double ry = p0.y - odom.pose.pose.position.y;
    const double forward_proj = rx * hx + ry * hy;
    if (forward_proj <= 0.0) return false;

    // 2) prekazka sa pohybuje priblizne v smere robotovho heading
    //    (cos uhlu > 0.5 => uhol < 60 deg)
    const double align = (ovx * hx + ovy * hy) / ospeed;
    return align > 0.5;
  }

  // ---- Temporal overlap check ------------------------------------------
  //
  // dynamic_obstacle_predictor publikuje /predicted_obstacle_path ako
  // nav_msgs::Path so vzorkovanim po `obstacle_prediction_dt_` (default
  // 0.1 s). Header.stamp je cas kedy bola predikcia spocitana, takze
  // pose[k] zodpoveda absolutnemu casu  stamp + k*obstacle_prediction_dt_.
  //
  // Tato funkcia forward-simuluje robotovu polohu pre kandidatne (v, w)
  // a v kazdom kroku robotovej predikcie najde zodpovedajuci index k v
  // obstacle path (matchovany na absolutny cas). Ak vzdialenost medzi
  // robot[i] a obstacle[k] padne pod temporal_safety_distance_, vraciame
  // prvy taky konflikt.
  //
  // Stale paths su odfiltrovane cez temporal_path_max_age_.
  struct TemporalConflict
  {
    bool   present    = false;
    int    step       = -1;     // index v robotovej predikcii (1..n)
    double t_conflict = 0.0;    // sekundy od teraz
    double dist       = 0.0;    // skutocna vzdialenost v bode kolizie
  };

  TemporalConflict checkTemporalConflict(const nav_msgs::Odometry& odom,
                                         const nav_msgs::Path& obs_path,
                                         double v, double w) const
  {
    TemporalConflict tc;
    if (obs_path.poses.empty())
      return tc;

    const ros::Time now = ros::Time::now();
    const double age = (now - obs_path.header.stamp).toSec();
    if (age > temporal_path_max_age_)
      return tc;

    const double yaw0 = yawFromQuaternion(odom.pose.pose.orientation);
    double cx = odom.pose.pose.position.x;
    double cy = odom.pose.pose.position.y;
    double cyaw = yaw0;

    const int n_steps = static_cast<int>(std::ceil(temporal_prediction_time_ / prediction_dt_));
    const double base_offset = age;  // (now - path.stamp) v sekundach

    const int n_obs = static_cast<int>(obs_path.poses.size());
    for (int i = 1; i <= n_steps; ++i)
    {
      cyaw += w * prediction_dt_;
      cx   += v * std::cos(cyaw) * prediction_dt_;
      cy   += v * std::sin(cyaw) * prediction_dt_;

      const double t_abs = base_offset + static_cast<double>(i) * prediction_dt_;
      if (t_abs < 0.0)
        continue;
      int k = static_cast<int>(std::round(t_abs / obstacle_prediction_dt_));
      if (k < 0 || k >= n_obs)
        continue;

      const auto& op = obs_path.poses[k].pose.position;
      const double dx = cx - op.x;
      const double dy = cy - op.y;
      const double d = std::hypot(dx, dy);
      if (d < temporal_safety_distance_)
      {
        tc.present    = true;
        tc.step       = i;
        tc.t_conflict = static_cast<double>(i) * prediction_dt_;
        tc.dist       = d;
        return tc;
      }
    }
    return tc;
  }

  // ---- Lethal escape mode ------------------------------------------------
  //
  // Aktivuje sa ked robot pozicia padne do lethal bunky. Simuluje niekolko
  // kandidatov (v, w), vypocita najskorsi cas vystupu z lethal zony a
  // (ak mame obstacle prediction) aj minimalnu vzdialenost od obstacle
  // predikcie v temporalne zarovnanych krokoch. Vyberie kandidata, ktory
  // najskor unika, s penalizaciou ak sa blizi k prekazke.
  EscapeScore evaluateEscape(const nav_msgs::Odometry& odom,
                              const nav_msgs::OccupancyGrid& cm,
                              const nav_msgs::Path* obs_path,
                              double v, double w) const
  {
    EscapeScore s;

    const double yaw0 = yawFromQuaternion(odom.pose.pose.orientation);
    double cx = odom.pose.pose.position.x;
    double cy = odom.pose.pose.position.y;
    double cyaw = yaw0;

    const int n_steps = static_cast<int>(std::ceil(escape_horizon_ / prediction_dt_));

    double obs_age = 0.0;
    int n_obs = 0;
    bool obs_valid = false;
    if (obs_path != nullptr && !obs_path->poses.empty())
    {
      obs_age = (ros::Time::now() - obs_path->header.stamp).toSec();
      n_obs = static_cast<int>(obs_path->poses.size());
      obs_valid = (obs_age <= temporal_path_max_age_);
    }

    for (int i = 1; i <= n_steps; ++i)
    {
      cyaw += w * prediction_dt_;
      cx   += v * std::cos(cyaw) * prediction_dt_;
      cy   += v * std::sin(cyaw) * prediction_dt_;

      if (!s.feasible && !cellLethal(cm, cx, cy))
      {
        s.feasible = true;
        s.t_exit = static_cast<double>(i) * prediction_dt_;
      }

      if (obs_valid)
      {
        const double t_abs = obs_age + static_cast<double>(i) * prediction_dt_;
        int k = static_cast<int>(std::round(t_abs / obstacle_prediction_dt_));
        if (k >= 0 && k < n_obs)
        {
          const auto& op = obs_path->poses[k].pose.position;
          const double d = std::hypot(cx - op.x, cy - op.y);
          s.min_obs_dist = std::min(s.min_obs_dist, d);
        }
      }
    }
    return s;
  }

  bool selectEscape(const nav_msgs::Odometry& odom,
                    const nav_msgs::OccupancyGrid& cm,
                    const nav_msgs::Path* obs_path,
                    geometry_msgs::Twist& out,
                    const char*& mode_out,
                    std::string& scores_out) const
  {
    const EscapeCandidate candidates[] = {
        { +escape_v_,  0.0,          "ESC_FWD"   },
        { -escape_v_,  0.0,          "ESC_REV"   },
        { +escape_v_, +escape_w_,    "ESC_FWD_L" },
        { +escape_v_, -escape_w_,    "ESC_FWD_R" },
        { -escape_v_, +escape_w_,    "ESC_REV_L" },
        { -escape_v_, -escape_w_,    "ESC_REV_R" },
        {  0.0,       +escape_w_,    "ESC_ROT_L" },
        {  0.0,       -escape_w_,    "ESC_ROT_R" },
    };

    int best_idx = -1;
    double best_score = std::numeric_limits<double>::infinity();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    for (std::size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i)
    {
      const EscapeScore s = evaluateEscape(odom, cm, obs_path,
                                           candidates[i].v, candidates[i].w);

      oss << " " << candidates[i].name << "[";
      if (!s.feasible) {
        oss << "BLOCKED]";
        continue;
      }

      // Skor: nizsi t_exit je lepsi; kandidat, ktorym v temporalnom
      // matchingu preprvavame prekazku do bezpecnej vzdialenosti, dostane
      // vysoku penalizaciu (nechceme vbehnut priamo pod kocku).
      double score = s.t_exit;
      const bool penalized = (s.min_obs_dist < temporal_safety_distance_);
      if (penalized)
        score += 10.0;

      // Secondary: preferuj trajektorie, ktore zostanu dalej od predikcie
      // prekazky. Pri rovnakom t_exit to rozbije tie medzi REV / REV_L / REV_R.
      // Tento bonus nemoze preklopit FWD vs REV poradie, lebo BLOCKED
      // kandidati nedostanu skore vobec.
      if (std::isfinite(s.min_obs_dist))
        score -= 0.01 * std::min(s.min_obs_dist, 3.0);

      oss << "t=" << s.t_exit
          << " d=" << (std::isfinite(s.min_obs_dist) ? s.min_obs_dist : -1.0)
          << " s=" << score
          << (penalized ? " !" : "")
          << "]";

      if (score < best_score)
      {
        best_score = score;
        best_idx = static_cast<int>(i);
      }
    }

    scores_out = oss.str();

    if (best_idx < 0)
      return false;

    out.linear.x = candidates[best_idx].v;
    out.linear.y = 0.0;
    out.angular.z = candidates[best_idx].w;
    mode_out = candidates[best_idx].name;
    return true;
  }

  // Forward-simuluje robota cez headon_horizon_ a meria min vzdialenost
  // od /predicted_obstacle_path v casovo zarovnanych krokoch.
  //
  // Feasibility: povodne sme blokovali ak ktorykolvek step bol v lethal,
  // ale pri vacsich rychlostiach / siroch arcov casto jeden neskorsi
  // cell v PredictedSweepRiskLayer corridor-e (cost ≥ 85) zablokoval
  // VSETKY REV_L/REV_R kandidatov a fallback vybral cistu reverze.
  //
  // Nova logika: blokujeme iba ak prvych `headon_min_feasible_steps_`
  // kokov je v lethal (bezprostredna kolizia — ani nezacneme manever).
  // Neskorsi lethal iba preruseni akumulaciu `min_obs_dist` — dalsi
  // cmd_vel tick (10 Hz) reevaluuje situaciu s novym stavom a vyberie
  // dalsi kandidat. Toto umoznuje manevru startnut aj ked arc neskor
  // preletí cez sweep corridor, ktory sa medzi tikmi hybe s prekazkou.
  HeadOnScore evaluateHeadOnCandidate(const nav_msgs::Odometry& odom,
                                       const nav_msgs::OccupancyGrid& cm,
                                       const nav_msgs::Path& obs_path,
                                       double v, double w) const
  {
    HeadOnScore s;
    s.feasible = false;

    const double yaw0 = yawFromQuaternion(odom.pose.pose.orientation);
    double cx = odom.pose.pose.position.x;
    double cy = odom.pose.pose.position.y;
    double cyaw = yaw0;

    const int n_steps = static_cast<int>(std::ceil(headon_horizon_ / prediction_dt_));
    const double obs_age = (ros::Time::now() - obs_path.header.stamp).toSec();
    const int n_obs = static_cast<int>(obs_path.poses.size());
    const bool obs_valid = (obs_age <= temporal_path_max_age_) && (n_obs > 0);

    for (int i = 1; i <= n_steps; ++i)
    {
      cyaw += w * prediction_dt_;
      cx   += v * std::cos(cyaw) * prediction_dt_;
      cy   += v * std::sin(cyaw) * prediction_dt_;

      const bool lethal_here = cellLethal(cm, cx, cy);

      if (lethal_here && i <= headon_min_feasible_steps_)
      {
        // Bezprostredna kolizia v prvych N krokoch → candidate truly blocked.
        return s;
      }

      // Prezili sme aspon prvy lethal-free step → candidate je feasible.
      // Dalsie lethal steps neblokuju, iba preruseni scoring.
      s.feasible = true;

      if (lethal_here)
        break;

      if (obs_valid)
      {
        const double t_abs = obs_age + static_cast<double>(i) * prediction_dt_;
        const int k = static_cast<int>(std::round(t_abs / obstacle_prediction_dt_));
        if (k >= 0 && k < n_obs)
        {
          const auto& op = obs_path.poses[k].pose.position;
          const double d = std::hypot(cx - op.x, cy - op.y);
          s.min_obs_dist = std::min(s.min_obs_dist, d);
        }
      }
    }
    return s;
  }

  // Vyber HEAD-ON manevru: dvojfazovy picker.
  // 1. faza: len SWERVE_L a SWERVE_R — ak je aspon jeden feasible,
  //    vyberie sa ten s vyssim min_obs_dist (+ strana-lock bonus).
  //    Zmysel: head-on sa ma PREDBEHNUT/OBIST, nie uniknut dozadu.
  // 2. faza (fallback): REV, REV_L, REV_R — iba ked sa swerve neda.
  // Strana-lock hysteréza: ak posledny pick bol L do headon_side_lock_time_
  // dozadu, k Lavym kandidatom pripocitame maly bonus, aby sme nepreskakovali.
  bool selectHeadOnManeuver(const nav_msgs::Odometry& odom,
                             const nav_msgs::OccupancyGrid& cm,
                             const nav_msgs::Path& obs_path,
                             geometry_msgs::Twist& out,
                             const char*& mode_out,
                             std::string& scores_out) const
  {
    // Primarna strategia: reverse + turn (cuvanie so zatacanim).
    // Robot cuvne a sucasne rotuje, cim sa odklona od head-on kurzu
    // a nasledny TEB plan dokaze prekazku obist zboku.
    const HeadOnCandidate rev_turn_candidates[] = {
        {  headon_rev_v_,+headon_w_,  +1, "HEADON_REV_L"    },
        {  headon_rev_v_,-headon_w_,  -1, "HEADON_REV_R"    },
    };
    // Fallback 1: cista reverze (bez rotacie), ak oba rev+turn su blocked.
    // Fallback 2: forward swerve, posledna moznost ked zadne varianty blokuje stena.
    const HeadOnCandidate fallback_candidates[] = {
        {  headon_rev_v_, 0.0,         0, "HEADON_REV"      },
        { +headon_v_,    +headon_w_,  +1, "HEADON_SWERVE_L" },
        { +headon_v_,    -headon_w_,  -1, "HEADON_SWERVE_R" },
    };

    const ros::Time now = ros::Time::now();
    const bool lock_active = (last_headon_side_ != 0) &&
                              !last_headon_time_.isZero() &&
                              ((now - last_headon_time_).toSec() < headon_side_lock_time_);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    auto score_group = [&](const HeadOnCandidate* arr, std::size_t n,
                            int& best_idx, double& best_score) {
      best_idx = -1;
      best_score = -std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < n; ++i)
      {
        // Hard lock: pocas side-lock okna uplne preskocime kandidatov opacnej
        // strany — nescorujeme, ani nekandidujeme. Zabranuje kmitaniu medzi
        // REV_L/REV_R ked su obe symetricky dobre a noise v min_obs_dist
        // by inak flippol stranu.
        if (headon_hard_lock_ && lock_active &&
            arr[i].side != 0 && arr[i].side != last_headon_side_)
        {
          oss << " " << arr[i].name << "[LOCKED]";
          continue;
        }

        const HeadOnScore s = evaluateHeadOnCandidate(odom, cm, obs_path,
                                                       arr[i].v, arr[i].w);
        oss << " " << arr[i].name << "[";
        if (!s.feasible)
        {
          oss << "BLOCKED]";
          continue;
        }
        double score = std::isfinite(s.min_obs_dist) ? s.min_obs_dist : 999.0;
        const bool boosted = (lock_active && arr[i].side == last_headon_side_);
        if (boosted) score += headon_side_lock_bonus_;
        oss << "d=" << (std::isfinite(s.min_obs_dist) ? s.min_obs_dist : -1.0)
            << " s=" << score
            << (boosted ? " *" : "")
            << "]";
        if (score > best_score)
        {
          best_score = score;
          best_idx = static_cast<int>(i);
        }
      }
    };

    // 1. faza: REV+turn (primarna strategia).
    int rev_turn_best = -1;
    double rev_turn_score = 0.0;
    score_group(rev_turn_candidates,
                sizeof(rev_turn_candidates) / sizeof(rev_turn_candidates[0]),
                rev_turn_best, rev_turn_score);

    const HeadOnCandidate* picked = nullptr;
    if (rev_turn_best >= 0)
    {
      picked = &rev_turn_candidates[rev_turn_best];
    }
    else
    {
      // 2. faza (fallback): REV straight + SWERVE.
      int fb_best = -1;
      double fb_score = 0.0;
      score_group(fallback_candidates,
                  sizeof(fallback_candidates) / sizeof(fallback_candidates[0]),
                  fb_best, fb_score);
      if (fb_best >= 0)
        picked = &fallback_candidates[fb_best];
    }

    scores_out = oss.str();

    if (!picked)
      return false;

    out.linear.x = picked->v;
    out.linear.y = 0.0;
    out.angular.z = picked->w;
    mode_out = picked->name;

    if (picked->side != 0)
    {
      last_headon_side_ = picked->side;
      last_headon_time_ = now;
    }
    return true;
  }

  bool isRearClear(const nav_msgs::OccupancyGrid& cm, const Point2D& base, double yaw) const
  {
    // Probe niekolko bodov za robotom (proti smeru yaw).
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    const double step = 0.10;
    const int n = std::max(1, static_cast<int>(std::ceil(reverse_clear_probe_distance_ / step)));
    for (int i = 1; i <= n; ++i)
    {
      const double d = static_cast<double>(i) * step;
      const double px = base.x - cy * d;
      const double py = base.y - sy * d;
      if (cellLethal(cm, px, py))
        return false;
    }
    return true;
  }

  static double yawFromQuaternion(const geometry_msgs::Quaternion& q)
  {
    tf2::Quaternion tq;
    tf2::fromMsg(q, tq);
    double r = 0.0;
    double p = 0.0;
    double y = 0.0;
    tf2::Matrix3x3(tq).getRPY(r, p, y);
    return y;
  }

  void publishEmptyPath(const std::string& frame_id)
  {
    if (!publish_prediction_path_)
      return;
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = frame_id;
    path_pub_.publish(path);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;

  ros::Subscriber cmd_sub_;
  ros::Subscriber odom_sub_;
  ros::Subscriber costmap_sub_;
  ros::Subscriber obstacle_path_sub_;
  ros::Publisher cmd_pub_;
  ros::Publisher path_pub_;
  ros::Publisher marker_pub_;
  ros::Timer viz_timer_;

  std::mutex mutex_;
  nav_msgs::Odometry latest_odom_;
  nav_msgs::OccupancyGrid latest_costmap_;
  nav_msgs::Path latest_obstacle_path_;
  bool has_odom_;
  bool has_costmap_;
  bool has_obstacle_path_ = false;

  std::string cmd_in_topic_;
  std::string cmd_out_topic_;
  std::string odom_topic_;
  std::string local_costmap_topic_;
  std::string obstacle_path_topic_;
  std::string prediction_path_topic_;
  std::string prediction_marker_topic_;

  double prediction_time_;
  double viz_rate_;
  double marker_line_width_;
  double marker_z_;
  double prediction_dt_;
  double slow_horizon_;
  double stop_horizon_;
  double reverse_horizon_;
  double min_check_linear_velocity_;
  double min_check_angular_velocity_;
  int lethal_threshold_;
  double reverse_velocity_;
  double reverse_clear_probe_distance_;
  bool publish_prediction_path_;

  // Temporal overlap parametre
  double temporal_safety_distance_;
  double temporal_path_max_age_;
  double obstacle_prediction_dt_;
  double temporal_prediction_time_;

  // Escape mode
  double escape_horizon_;
  double escape_v_;
  double escape_w_;

  // Head-on maneuver
  double headon_v_;
  double headon_w_;
  double headon_rev_v_;
  double headon_horizon_;
  double headon_side_lock_time_;
  double headon_side_lock_bonus_;
  bool   headon_hard_lock_;
  int    headon_min_feasible_steps_;
  // Strana-lock hysteréza: -1 = R, 0 = none, +1 = L
  mutable int    last_headon_side_ = 0;
  mutable ros::Time last_headon_time_;

  // Command latch (viď cmd_latch_time_ komentar pri param loadingu).
  double cmd_latch_time_;
  mutable ros::Time cmd_latch_until_;
  mutable geometry_msgs::Twist cmd_latch_value_;
  mutable std::string cmd_latch_reason_;
};

}  // namespace

int main(int argc, char** argv)
{
  ros::init(argc, argv, "robot_motion_predictor");
  RobotMotionPredictor node;
  ros::spin();
  return 0;
}
