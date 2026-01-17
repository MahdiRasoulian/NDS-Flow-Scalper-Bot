//+------------------------------------------------------------------+
//| NDS Flow Scalper Parity EA (Phase 0-1)                           |
//| Strict parity scaffolding for Python logic                       |
//+------------------------------------------------------------------+
#property strict
#property version   "0.1"

#include "../Include/NDS_BarSeries.mqh"
#include "../Include/NDS_SMC.mqh"

input string         InpSymbol            = "";
input ENUM_TIMEFRAMES InpTimeframe        = PERIOD_M15;
input int            InpMagicNumber       = 902410;
input double         InpMaxSpreadPoints   = 30.0;
input int            InpNYCooldownMinutes = 15;
input int            InpBreakerLookback   = 50;
input int            InpIFVGMaxBars       = 60;
input int            InpFreshBarsWindow   = 30;
input bool           InpAllowRetest       = false;
input int            InpMaxRetests        = 1;
input double         InpRiskPercent       = 0.5;
input double         InpSLPips            = 12.0;
input double         InpTP1Pips           = 10.0;
input double         InpTP2Pips           = 20.0;
input double         InpATRMultiplier     = 2.0;
input bool           InpDebug             = true;

CBarSeries      g_bars;
CLogger         g_logger;
CSMC_Analyzer   g_smc;
CFilters        g_filters;

string          g_symbol;

bool IsNewBar(const string symbol, ENUM_TIMEFRAMES tf, datetime &last_bar_time)
{
   MqlRates rates[2];
   ArraySetAsSeries(rates, true);
   if(CopyRates(symbol, tf, 0, 2, rates) != 2)
      return false;
   if(rates[0].time != last_bar_time)
   {
      last_bar_time = rates[0].time;
      return true;
   }
   return false;
}

void DrawZone(const string name, const datetime time1, const double price1, const datetime time2, const double price2)
{
   if(ObjectFind(0, name) >= 0)
      ObjectDelete(0, name);
   ObjectCreate(0, name, OBJ_RECTANGLE, 0, time1, price1, time2, price2);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clrSlateBlue);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
}

void DrawSignal(const string name, const datetime time1, const double price, const ENUM_SIGNAL signal, const string label)
{
   if(ObjectFind(0, name) >= 0)
      ObjectDelete(0, name);
   ObjectCreate(0, name, OBJ_ARROW, 0, time1, price);
   ObjectSetInteger(0, name, OBJPROP_ARROWCODE, signal == SIGNAL_BUY ? 233 : 234);
   ObjectSetInteger(0, name, OBJPROP_COLOR, signal == SIGNAL_BUY ? clrLimeGreen : clrTomato);

   string text_name = name + "_txt";
   if(ObjectFind(0, text_name) >= 0)
      ObjectDelete(0, text_name);
   ObjectCreate(0, text_name, OBJ_TEXT, 0, time1, price);
   ObjectSetString(0, text_name, OBJPROP_TEXT, label);
   ObjectSetInteger(0, text_name, OBJPROP_COLOR, clrWhite);
}

int OnInit()
{
   g_symbol = (InpSymbol == "") ? _Symbol : InpSymbol;
   g_logger.Enable(InpDebug);

   return INIT_SUCCEEDED;
}

void OnTick()
{
   static datetime last_bar_time = 0;
   if(!IsNewBar(g_symbol, InpTimeframe, last_bar_time))
      return;

   int lookback = MathMax(InpBreakerLookback, 5);
   if(!g_bars.Refresh(g_symbol, InpTimeframe, lookback + 5))
      return;

   MqlRates bar;
   MqlRates prev;
   if(!g_bars.GetBar(1, bar) || !g_bars.GetBar(2, prev))
      return;

   double recent_high = bar.high;
   double recent_low = bar.low;
   for(int i = 1; i <= InpBreakerLookback && i < g_bars.Count(); i++)
   {
      MqlRates scan_bar;
      if(!g_bars.GetBar(i, scan_bar))
         continue;
      if(scan_bar.high > recent_high)
         recent_high = scan_bar.high;
      if(scan_bar.low < recent_low)
         recent_low = scan_bar.low;
   }

   double spread_points = (double)SymbolInfoInteger(g_symbol, SYMBOL_SPREAD);

   g_logger.Log("[BAR]", StringFormat("time=%s close=%.5f", TimeToString(bar.time, TIME_DATE|TIME_MINUTES), bar.close));

   bool spread_ok = g_filters.SpreadOk(spread_points, InpMaxSpreadPoints, g_logger);
   bool session_ok = g_filters.SessionOk(bar.time, InpNYCooldownMinutes, g_logger);

   SignalResult signal;
   g_smc.Analyze(bar, prev, recent_high, recent_low, SymbolInfoDouble(g_symbol, SYMBOL_POINT) * 2.0, signal, g_logger);

   if(spread_ok && session_ok && signal.signal != SIGNAL_NONE)
   {
      string reasons = "";
      for(int i = 0; i < signal.reason_count; i++)
      {
         if(i > 0)
            reasons += ",";
         reasons += signal.reasons[i];
      }
      g_logger.Log("[SIGNAL]", StringFormat("signal=%s reasons=%s zone=%s", signal.signal == SIGNAL_BUY ? "BUY" : "SELL",
                                            reasons, signal.zone_id));

      if(signal.zone_valid)
      {
         DrawZone("nds_zone_" + signal.zone_id, bar.time, signal.zone_high, prev.time, signal.zone_low);
      }
      DrawSignal("nds_signal_" + signal.zone_id, bar.time, bar.close, signal.signal, reasons);
   }
   else
   {
      g_logger.Log("[SIGNAL]", "signal=NONE reasons=filter_blocked");
   }
}
