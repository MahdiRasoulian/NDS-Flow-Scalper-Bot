#pragma once
#include <Arrays/ArrayString.mqh>

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
   CLogger *m_logger;

   bool DetectBreaker(const MqlRates &bar, const double recent_high, const double recent_low, const double sweep_buffer)
   {
      bool swept_high = bar.high > recent_high + sweep_buffer;
      bool swept_low  = bar.low < recent_low - sweep_buffer;
      return (swept_high || swept_low);
   }

   bool DetectIFVG(const MqlRates &bar, const MqlRates &prev_bar, double &gap_low, double &gap_high)
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
   CSMC_Analyzer() : m_logger(NULL) {}

   void SetLogger(CLogger *logger)
   {
      m_logger = logger;
   }

   SignalResult Analyze(const MqlRates &bar, const MqlRates &prev_bar, const double recent_high,
                        const double recent_low, const double sweep_buffer)
   {
      SignalResult result;
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

      if(m_logger != NULL)
      {
         m_logger.Log("[SMC]", StringFormat("breaker_found=%s ifvg_found=%s", breaker_found ? "true" : "false",
                                            ifvg_found ? "true" : "false"));
      }

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

      return result;
   }
};

class CMomentum
{
private:
   int m_adx_handle;
   double m_adx_buffer[];

public:
   CMomentum() : m_adx_handle(INVALID_HANDLE) {}

   bool Init(const string symbol, ENUM_TIMEFRAMES tf, const int period)
   {
      m_adx_handle = iADX(symbol, tf, period);
      ArraySetAsSeries(m_adx_buffer, true);
      return (m_adx_handle != INVALID_HANDLE);
   }

   bool Update()
   {
      if(m_adx_handle == INVALID_HANDLE)
         return false;
      int copied = CopyBuffer(m_adx_handle, 0, 0, 2, m_adx_buffer);
      return (copied > 0);
   }

   double Value() const
   {
      if(ArraySize(m_adx_buffer) == 0)
         return 0.0;
      return m_adx_buffer[0];
   }
};

class CFilters
{
private:
   CLogger *m_logger;

public:
   CFilters() : m_logger(NULL) {}

   void SetLogger(CLogger *logger)
   {
      m_logger = logger;
   }

   bool SpreadOk(const double spread_points, const double max_spread_points)
   {
      bool ok = (spread_points <= max_spread_points);
      if(m_logger != NULL)
      {
         m_logger->Log("[FILTER]", StringFormat("spread_ok=%s spread=%.1f max=%.1f", ok ? "true" : "false",
                                                spread_points, max_spread_points));
      }
      return ok;
   }

   bool SessionOk(const datetime bar_time, const int cooldown_minutes)
   {
      MqlDateTime dt;
      TimeToStruct(bar_time, dt);
      int minutes = dt.hour * 60 + dt.min;
      int ny_open = 13 * 60 + 30; // placeholder for NY open 13:30 UTC
      bool ok = (MathAbs(minutes - ny_open) > cooldown_minutes);
      if(m_logger != NULL)
      {
         m_logger->Log("[FILTER]", StringFormat("session_ok=%s cooldown_minutes=%d", ok ? "true" : "false",
                                                cooldown_minutes));
      }
      return ok;
   }
};
