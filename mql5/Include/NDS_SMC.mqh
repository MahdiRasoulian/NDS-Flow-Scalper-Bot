#pragma once

enum ENUM_SIGNAL
{
   SIGNAL_NONE = 0,
   SIGNAL_BUY  = 1,
   SIGNAL_SELL = -1
};

struct SignalResult
{
   ENUM_SIGNAL signal;
   double      confidence;
   string      reasons[10];
   int         reason_count;
   string      zone_id;
   string      zone_type;
   double      zone_low;
   double      zone_high;
   bool        zone_valid;
   bool        zone_fresh;
   int         zone_retests;
};

struct ZoneState
{
   string id;
   string type;
   double low;
   double high;
   datetime created_time;
   int retests;
};

class CLogger
{
private:
   bool m_enabled;

public:
   CLogger() : m_enabled(true) {}

   void Enable(const bool enabled)
   {
      m_enabled = enabled;
   }

   void Log(const string tag, const string message)
   {
      if(!m_enabled)
         return;
      Print("[NDS]", tag, " ", message);
   }
};

class CSMC_Analyzer
{
private:
   bool DetectBreaker(MqlRates &bar, const double recent_high, const double recent_low, const double sweep_buffer)
   {
      bool swept_high = bar.high > recent_high + sweep_buffer;
      bool swept_low  = bar.low < recent_low - sweep_buffer;
      return (swept_high || swept_low);
   }

   bool DetectIFVG(MqlRates &bar, MqlRates &prev_bar, double &gap_low, double &gap_high)
   {
      if(bar.low > prev_bar.high)
      {
         gap_low = prev_bar.high;
         gap_high = bar.low;
         return true;
      }
      if(bar.high < prev_bar.low)
      {
         gap_low = bar.high;
         gap_high = prev_bar.low;
         return true;
      }
      return false;
   }

public:
   CSMC_Analyzer() {}

   void Analyze(MqlRates &bar, MqlRates &prev_bar, const double recent_high,
                const double recent_low, const double sweep_buffer, SignalResult &result, CLogger &logger)
   {
      result.signal = SIGNAL_NONE;
      result.confidence = 0.0;
      result.reason_count = 0;
      result.zone_id = "";
      result.zone_type = "";
      result.zone_low = 0.0;
      result.zone_high = 0.0;
      result.zone_valid = false;
      result.zone_fresh = false;
      result.zone_retests = 0;

      double gap_low = 0.0;
      double gap_high = 0.0;
      bool breaker_found = DetectBreaker(bar, recent_high, recent_low, sweep_buffer);
      bool ifvg_found = DetectIFVG(bar, prev_bar, gap_low, gap_high);

      logger.Log("[SMC]", StringFormat("breaker_found=%s ifvg_found=%s",
                                       breaker_found ? "true" : "false",
                                       ifvg_found ? "true" : "false"));

      if(breaker_found && ifvg_found)
      {
         result.signal = (bar.close > bar.open) ? SIGNAL_BUY : SIGNAL_SELL;
         result.confidence = 1.0;
         result.reasons[result.reason_count++] = "breaker";
         result.reasons[result.reason_count++] = "ifvg";
         result.zone_valid = true;
         result.zone_fresh = true;
         result.zone_low = gap_low;
         result.zone_high = gap_high;
         result.zone_type = "IFVG";
         result.zone_id = StringFormat("ifvg_%I64d", (long)bar.time);
      }
   }
};

class CFilters
{
public:
   CFilters() {}

   bool SpreadOk(const double spread_points, const double max_spread_points, CLogger &logger)
   {
      bool ok = (spread_points <= max_spread_points);
      logger.Log("[FILTER]", StringFormat("spread_ok=%s spread=%.1f max=%.1f",
                                          ok ? "true" : "false",
                                          spread_points, max_spread_points));
      return ok;
   }

   bool SessionOk(const datetime bar_time, const int cooldown_minutes, CLogger &logger)
   {
      MqlDateTime dt;
      TimeToStruct(bar_time, dt);
      int minutes = dt.hour * 60 + dt.min;
      int ny_open = 13 * 60 + 30;
      bool ok = (MathAbs(minutes - ny_open) > cooldown_minutes);
      logger.Log("[FILTER]", StringFormat("session_ok=%s cooldown_minutes=%d",
                                          ok ? "true" : "false",
                                          cooldown_minutes));
      return ok;
   }
};
