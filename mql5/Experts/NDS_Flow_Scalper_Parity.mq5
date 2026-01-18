#property copyright "NDS Flow Scalper"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

input group "== Trading Rules (from bot_config) =="
input int    InpMaxPositions              = 5;
input int    InpMinCandlesBetweenTrades   = 4;
input double InpMinRiskReward             = 0.55;

input group "== Risk Management =="
input double InpRiskPercent               = 1.2;
input double InpMaxDailyRiskPercent       = 6.0;
input double InpSLPips                    = 15.0;
input double InpTP1Pips                   = 35.0;
input bool   InpTP2Enabled                = true;
input double InpTP2Pips                   = 70.0;
input double InpMaxSpreadPips             = 2.5;

input group "== SMC & Flow Logic =="
input int    InpBreakerLookback           = 50;
input double InpBreakerDisplacementATR    = 1.2;
input double InpBrkDisplacementBodyRatio  = 0.25;
input int    InpIFVGMaxBars               = 60;
input int    InpSMCMaxZoneAge             = 240;

input group "== Indicators & Sessions =="
input ENUM_TIMEFRAMES InpSignalTimeframe  = PERIOD_M15;
input int    InpADXPeriod                 = 14;
input double InpADXStrong                 = 25.0;
input int    InpATRPeriod                 = 14;
input double InpATRSpikeMult              = 3.0;
input int    InpNYOpenHour                = 15;
input int    InpNYOpenMinute              = 0;
input int    InpNYOpenCooldownMinutes     = 15;

input group "== Trade Management =="
input double InpFlowTp1PartialClosePct    = 0.5;
input bool   InpFlowTp1MoveSLtoBE         = true;
input double InpTrailingATRMult           = 2.0;

enum ENDSZoneType
{
   ZONE_BREAKER = 1,
   ZONE_IFVG = 2
};

enum ENDSDirection
{
   DIR_BULLISH = 1,
   DIR_BEARISH = -1
};

struct SZone
{
   string id;
   datetime created;
   double high;
   double low;
   ENDSDirection direction;
   ENDSZoneType type;
   bool fresh;
   int bars_alive;
   bool touched;
};

struct SSignal
{
   bool valid;
   string reason;
   ENDSDirection direction;
   double entry_price;
   SZone zone;
};

struct SPositionState
{
   ulong ticket;
   bool tp1_hit;
   double tp1_price;
   double tp2_price;
};

double GetATR(int shift);
double GetADX(int shift);
int CountOpenPositions();
double CalcOpenRiskPercent();
double PipsToPrice(double pips);

class CSignalEngine
{
private:
   SZone m_zones[64];
   int m_zone_count;
   double m_atr;
   double m_adx;

   double PipSize()
   {
      return PipsToPrice(1.0);
   }

   void DrawZone(const SZone &zone)
   {
      if(ObjectFind(0, zone.id) >= 0)
         return;
      ObjectCreate(0, zone.id, OBJ_RECTANGLE, 0, zone.created, zone.high, TimeCurrent(), zone.low);
      color fill = zone.type == ZONE_BREAKER ? clrDeepSkyBlue : clrDarkOrange;
      ObjectSetInteger(0, zone.id, OBJPROP_COLOR, fill);
      ObjectSetInteger(0, zone.id, OBJPROP_BACK, true);
      ObjectSetInteger(0, zone.id, OBJPROP_STYLE, STYLE_SOLID);
      ObjectSetInteger(0, zone.id, OBJPROP_WIDTH, 1);
   }

   void DrawEntryArrow(const string &id, datetime when, double price, ENDSDirection direction)
   {
      if(ObjectFind(0, id) >= 0)
         return;
      ObjectCreate(0, id, OBJ_ARROW, 0, when, price);
      ObjectSetInteger(0, id, OBJPROP_COLOR, direction == DIR_BULLISH ? clrLime : clrTomato);
      ObjectSetInteger(0, id, OBJPROP_ARROWCODE, direction == DIR_BULLISH ? 233 : 234);
      ObjectSetInteger(0, id, OBJPROP_WIDTH, 2);
   }

   bool HasSweep(const MqlRates &rates[], int lookback, ENDSDirection direction)
   {
      double sweep_price = 0.0;
      if(direction == DIR_BULLISH)
      {
         sweep_price = rates[1].low;
         for(int i = 2; i <= lookback && i < ArraySize(rates); i++)
         {
            if(rates[i].low < sweep_price)
               return true;
         }
      }
      else
      {
         sweep_price = rates[1].high;
         for(int i = 2; i <= lookback && i < ArraySize(rates); i++)
         {
            if(rates[i].high > sweep_price)
               return true;
         }
      }
      return false;
   }

   bool DisplacementConfirmed(const MqlRates &rates[], double atr, ENDSDirection direction)
   {
      double body = MathAbs(rates[1].close - rates[1].open);
      if(body < atr * InpBreakerDisplacementATR)
         return false;
      if(rates[1].high - rates[1].low > 0.0)
      {
         double body_ratio = body / (rates[1].high - rates[1].low);
         if(body_ratio < InpBrkDisplacementBodyRatio)
            return false;
      }
      if(direction == DIR_BULLISH && rates[1].close <= rates[1].open)
         return false;
      if(direction == DIR_BEARISH && rates[1].close >= rates[1].open)
         return false;
      return true;
   }

   bool FindBreaker(const MqlRates &rates[], ENDSDirection direction, SZone &out_zone)
   {
      if(!HasSweep(rates, InpBreakerLookback, direction))
         return false;

      for(int i = 2; i <= InpBreakerLookback && i < ArraySize(rates); i++)
      {
         bool is_ob = direction == DIR_BULLISH ? (rates[i].close < rates[i].open) : (rates[i].close > rates[i].open);
         if(!is_ob)
            continue;

         double ob_high = rates[i].high;
         double ob_low = rates[i].low;
         bool mitigated = false;
         for(int j = i - 1; j >= 1; j--)
         {
            if(direction == DIR_BULLISH && rates[j].low <= ob_low)
            {
               mitigated = true;
               break;
            }
            if(direction == DIR_BEARISH && rates[j].high >= ob_high)
            {
               mitigated = true;
               break;
            }
         }
         if(!mitigated)
            continue;

         if(!DisplacementConfirmed(rates, m_atr, direction))
            continue;

         if(direction == DIR_BULLISH && rates[1].close <= ob_high)
            continue;
         if(direction == DIR_BEARISH && rates[1].close >= ob_low)
            continue;

         out_zone.id = StringFormat("BREAKER_%s_%d", direction == DIR_BULLISH ? "BUY" : "SELL", (int)rates[1].time);
         out_zone.created = rates[1].time;
         out_zone.high = ob_high;
         out_zone.low = ob_low;
         out_zone.direction = direction;
         out_zone.type = ZONE_BREAKER;
         out_zone.fresh = true;
         out_zone.touched = false;
         out_zone.bars_alive = 0;
         return true;
      }
      return false;
   }

   bool FindIFVGInversion(const MqlRates &rates[], ENDSDirection direction, SZone &out_zone)
   {
      if(ArraySize(rates) < 4)
         return false;

      MqlRates r1 = rates[1];
      MqlRates r3 = rates[3];

      if(direction == DIR_BULLISH)
      {
         if(r1.low <= r3.high)
            return false;
         double gap_low = r3.high;
         double gap_high = r1.low;
         if(r1.close >= gap_low)
            return false;
         out_zone.id = StringFormat("IFVG_BUY_%d", (int)r1.time);
         out_zone.created = r1.time;
         out_zone.high = gap_high;
         out_zone.low = gap_low;
         out_zone.direction = DIR_BULLISH;
         out_zone.type = ZONE_IFVG;
         out_zone.fresh = true;
         out_zone.touched = false;
         out_zone.bars_alive = 0;
         return true;
      }
      else
      {
         if(r1.high >= r3.low)
            return false;
         double gap_high = r3.low;
         double gap_low = r1.high;
         if(r1.close <= gap_high)
            return false;
         out_zone.id = StringFormat("IFVG_SELL_%d", (int)r1.time);
         out_zone.created = r1.time;
         out_zone.high = gap_high;
         out_zone.low = gap_low;
         out_zone.direction = DIR_BEARISH;
         out_zone.type = ZONE_IFVG;
         out_zone.fresh = true;
         out_zone.touched = false;
         out_zone.bars_alive = 0;
         return true;
      }
      return false;
   }

   void AddZone(const SZone &zone)
   {
      if(m_zone_count >= ArraySize(m_zones))
         return;
      m_zones[m_zone_count] = zone;
      m_zone_count++;
      DrawZone(zone);
   }

public:
   CSignalEngine() : m_zone_count(0), m_atr(0.0), m_adx(0.0) {}

   void UpdateIndicators()
   {
      m_atr = GetATR(1);
      m_adx = GetADX(1);
   }

   void Analyze()
   {
      MqlRates rates[64];
      int copied = CopyRates(_Symbol, InpSignalTimeframe, 0, 64, rates);
      if(copied < 10)
         return;

      m_zone_count = 0;
      UpdateIndicators();

      SZone zone;
      if(FindBreaker(rates, DIR_BULLISH, zone))
         AddZone(zone);
      if(FindBreaker(rates, DIR_BEARISH, zone))
         AddZone(zone);
      if(FindIFVGInversion(rates, DIR_BULLISH, zone))
         AddZone(zone);
      if(FindIFVGInversion(rates, DIR_BEARISH, zone))
         AddZone(zone);

      for(int i = 0; i < m_zone_count; i++)
      {
         m_zones[i].bars_alive++;
         if(m_zones[i].type == ZONE_IFVG && m_zones[i].bars_alive > InpIFVGMaxBars)
            m_zones[i].fresh = false;
         if(m_zones[i].bars_alive > InpSMCMaxZoneAge)
            m_zones[i].fresh = false;
      }
   }

   SSignal GenerateSignal()
   {
      SSignal signal;
      signal.valid = false;
      signal.reason = "";
      signal.direction = DIR_BULLISH;
      signal.entry_price = 0.0;

      MqlRates rates[8];
      int copied = CopyRates(_Symbol, InpSignalTimeframe, 0, 8, rates);
      if(copied < 4)
         return signal;

      UpdateIndicators();

      double price = rates[1].close;
      for(int i = 0; i < m_zone_count; i++)
      {
         if(!m_zones[i].fresh)
            continue;
         if(m_zones[i].type == ZONE_IFVG && m_zones[i].bars_alive > InpIFVGMaxBars)
            continue;
         if(price >= m_zones[i].low && price <= m_zones[i].high)
         {
            signal.valid = true;
            signal.direction = m_zones[i].direction;
            signal.entry_price = price;
            signal.reason = m_zones[i].type == ZONE_BREAKER ? "BREAKER" : "IFVG";
            signal.zone = m_zones[i];
            m_zones[i].touched = true;
            m_zones[i].fresh = false;
            return signal;
         }
      }

      double body = MathAbs(rates[1].close - rates[1].open);
      double momentum_ratio = m_atr > 0.0 ? body / m_atr : 0.0;
      if(m_adx >= InpADXStrong && momentum_ratio >= InpBrkDisplacementBodyRatio)
      {
         signal.valid = true;
         signal.direction = rates[1].close > rates[1].open ? DIR_BULLISH : DIR_BEARISH;
         signal.entry_price = price;
         signal.reason = "MOMENTUM";
      }

      return signal;
   }

   void DrawSignal(const SSignal &signal)
   {
      if(!signal.valid)
         return;
      string id = StringFormat("ENTRY_%s_%d", signal.reason, (int)TimeCurrent());
      DrawEntryArrow(id, TimeCurrent(), signal.entry_price, signal.direction);
   }
};

class CRiskManager
{
public:
   double PipSize()
   {
      return PipsToPrice(1.0);
   }

   double NormalizePips(double pips)
   {
      if(pips < 1.0)
         return 1.0;
      return pips;
   }

   void BuildLevels(ENDSDirection direction, double entry, double sl_pips, double tp1_pips, double tp2_pips,
                    double &sl, double &tp1, double &tp2)
   {
      double sl_dist = PipsToPrice(NormalizePips(sl_pips));
      double tp1_dist = PipsToPrice(NormalizePips(tp1_pips));
      double tp2_dist = PipsToPrice(NormalizePips(tp2_pips));
      double stop_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double spread = ask - bid;

      if(direction == DIR_BULLISH)
      {
         sl = entry - sl_dist;
         tp1 = entry + tp1_dist;
         tp2 = entry + tp2_dist;
      }
      else
      {
         sl = entry + sl_dist;
         tp1 = entry - tp1_dist;
         tp2 = entry - tp2_dist;
      }

      if(spread > 0.0)
      {
         if(direction == DIR_BULLISH && (entry - sl) < spread)
            sl = entry - spread;
         if(direction == DIR_BEARISH && (sl - entry) < spread)
            sl = entry + spread;
      }

      if(stop_level > 0.0)
      {
         if(direction == DIR_BULLISH && (entry - sl) < stop_level)
            sl = entry - stop_level;
         if(direction == DIR_BEARISH && (sl - entry) < stop_level)
            sl = entry + stop_level;
      }

      sl = NormalizeDouble(sl, _Digits);
      tp1 = NormalizeDouble(tp1, _Digits);
      tp2 = NormalizeDouble(tp2, _Digits);
   }

   double CalcVolume(double sl_pips, double risk_percent, ENUM_ORDER_TYPE order_type, double entry_price)
   {
      double risk_money = AccountInfoDouble(ACCOUNT_BALANCE) * (risk_percent / 100.0);
      double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double pip_value = (tick_value / tick_size) * PipSize();
      if(pip_value <= 0.0)
         return 0.0;
      double volume = risk_money / (sl_pips * pip_value);
      double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
      double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
      double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
      volume = MathMax(min_lot, MathMin(max_lot, volume));
      volume = MathFloor(volume / step) * step;
      double free_margin = AccountInfoDouble(ACCOUNT_FREEMARGIN);
      double required_margin = 0.0;
      if(OrderCalcMargin(order_type, _Symbol, volume, entry_price, required_margin))
      {
         if(required_margin > free_margin * 0.8)
         {
            double scale = (free_margin * 0.8) / required_margin;
            volume = MathFloor((volume * scale) / step) * step;
         }
      }
      if(volume < min_lot)
         volume = 0.0;
      return volume;
   }
};

class CTradeExecutor
{
private:
   CTrade m_trade;
   SPositionState m_state;

public:
   CTradeExecutor()
   {
      m_state.ticket = 0;
      m_state.tp1_hit = false;
      m_state.tp1_price = 0.0;
      m_state.tp2_price = 0.0;
   }

   bool HasPosition()
   {
      return PositionSelect(_Symbol);
   }

   void ResetStateIfClosed()
   {
      if(!HasPosition())
      {
         m_state.ticket = 0;
         m_state.tp1_hit = false;
         m_state.tp1_price = 0.0;
      }
   }

   bool OpenPosition(const SSignal &signal, double sl, double tp1, double tp2, double volume)
   {
      bool result = false;
      double normalized_sl = NormalizeDouble(sl, _Digits);
      m_trade.SetDeviationInPoints(10);
      if(signal.direction == DIR_BULLISH)
         result = m_trade.Buy(volume, _Symbol, 0.0, normalized_sl, 0.0, signal.reason);
      else
         result = m_trade.Sell(volume, _Symbol, 0.0, normalized_sl, 0.0, signal.reason);

      if(result)
      {
         m_state.ticket = (ulong)m_trade.ResultOrder();
         m_state.tp1_hit = false;
         m_state.tp1_price = tp1;
         m_state.tp2_price = tp2;
         Print("TRADE_OPENED: ", signal.reason, " ticket=", (string)m_state.ticket, " volume=", DoubleToString(volume, 2));
      }
      else
      {
         Print("TRADE_REJECTED: ", signal.reason,
               " retcode=", (string)m_trade.ResultRetcode(),
               " desc=", m_trade.ResultRetcodeDescription());
      }
      return result;
   }

   void ManagePosition()
   {
      if(!HasPosition())
         return;

      ulong ticket = (ulong)PositionGetInteger(POSITION_TICKET);
      double price_open = PositionGetDouble(POSITION_PRICE_OPEN);
      double volume = PositionGetDouble(POSITION_VOLUME);
      double sl = PositionGetDouble(POSITION_SL);
      double tp1 = m_state.tp1_price;
      long type = PositionGetInteger(POSITION_TYPE);
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double current_price = type == POSITION_TYPE_BUY ? bid : ask;

      if(m_state.ticket != ticket)
      {
         m_state.ticket = ticket;
         m_state.tp1_hit = false;
      }

      if(!m_state.tp1_hit)
      {
         bool reached = (type == POSITION_TYPE_BUY && current_price >= tp1) ||
                        (type == POSITION_TYPE_SELL && current_price <= tp1);
         if(reached)
         {
            double close_volume = volume * InpFlowTp1PartialClosePct;
            double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
            if(close_volume < min_lot)
               close_volume = volume;
            if(!m_trade.PositionClosePartial(_Symbol, close_volume))
               Print("TP1_CLOSE_FAILED: retcode=", (string)m_trade.ResultRetcode(),
                     " desc=", m_trade.ResultRetcodeDescription());
            double breakeven = NormalizeDouble(price_open, _Digits);
            if(InpFlowTp1MoveSLtoBE)
            {
               if(!m_trade.PositionModify(_Symbol, breakeven, 0.0))
                  Print("BREAKEVEN_FAILED: retcode=", (string)m_trade.ResultRetcode(),
                        " desc=", m_trade.ResultRetcodeDescription());
            }
            m_state.tp1_hit = true;
            Print("TP1_HIT: ticket=", (string)ticket, " close_volume=", DoubleToString(close_volume, 2));
         }
      }

      if(m_state.tp1_hit)
      {
         if(InpTP2Enabled && m_state.tp2_price > 0.0)
         {
            bool tp2_reached = (type == POSITION_TYPE_BUY && current_price >= m_state.tp2_price) ||
                               (type == POSITION_TYPE_SELL && current_price <= m_state.tp2_price);
            if(tp2_reached)
            {
               if(!m_trade.PositionClose(_Symbol))
                  Print("TP2_CLOSE_FAILED: retcode=", (string)m_trade.ResultRetcode(),
                        " desc=", m_trade.ResultRetcodeDescription());
               else
                  Print("TP2_HIT: ticket=", (string)ticket);
               return;
            }
         }
         double atr = GetATR(1);
         if(atr <= 0.0)
            return;
         double trailing = atr * InpTrailingATRMult;
         if(type == POSITION_TYPE_BUY)
         {
            double new_sl = NormalizeDouble(current_price - trailing, _Digits);
            if(new_sl > sl)
            {
               if(!m_trade.PositionModify(_Symbol, new_sl, 0.0))
                  Print("TRAIL_MODIFY_FAILED: retcode=", (string)m_trade.ResultRetcode(),
                        " desc=", m_trade.ResultRetcodeDescription());
            }
         }
         else
         {
            double new_sl = NormalizeDouble(current_price + trailing, _Digits);
            if(new_sl < sl || sl == 0.0)
            {
               if(!m_trade.PositionModify(_Symbol, new_sl, 0.0))
                  Print("TRAIL_MODIFY_FAILED: retcode=", (string)m_trade.ResultRetcode(),
                        " desc=", m_trade.ResultRetcodeDescription());
            }
         }
      }
   }
};

CSignalEngine g_signal;
CRiskManager g_risk;
CTradeExecutor g_trader;

datetime g_last_bar_time = 0;
int g_atr_handle = INVALID_HANDLE;
int g_adx_handle = INVALID_HANDLE;
datetime g_last_trade_bar_time = 0;

double GetATR(int shift)
{
   if(g_atr_handle == INVALID_HANDLE)
      return 0.0;
   double buffer[2];
   if(CopyBuffer(g_atr_handle, 0, shift, 1, buffer) != 1)
      return 0.0;
   return buffer[0];
}

double GetADX(int shift)
{
   if(g_adx_handle == INVALID_HANDLE)
      return 0.0;
   double buffer[2];
   if(CopyBuffer(g_adx_handle, 0, shift, 1, buffer) != 1)
      return 0.0;
   return buffer[0];
}

bool IsNewCandle()
{
   datetime current = iTime(_Symbol, InpSignalTimeframe, 0);
   if(current != g_last_bar_time)
   {
      g_last_bar_time = current;
      return true;
   }
   return false;
}

bool PassFilters()
{
   double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double max_spread = PipsToPrice(InpMaxSpreadPips);
   if(spread > max_spread)
   {
      Print("FILTER_REJECT: SPREAD too high spread_price=", DoubleToString(spread, _Digits),
            " max_price=", DoubleToString(max_spread, _Digits));
      return false;
   }

   datetime now = TimeCurrent();
   MqlDateTime tm;
   TimeToStruct(now, tm);
   int ny_minutes = tm.hour * 60 + tm.min;
   int open_minutes = InpNYOpenHour * 60 + InpNYOpenMinute;
   if(MathAbs(ny_minutes - open_minutes) <= InpNYOpenCooldownMinutes)
   {
      Print("FILTER_REJECT: NY_COOLDOWN now=", (string)ny_minutes,
            " open=", (string)open_minutes, " window=", (string)InpNYOpenCooldownMinutes);
      return false;
   }

   double atr = GetATR(1);
   double atr_avg = 0.0;
   for(int i = 1; i <= 20; i++)
      atr_avg += GetATR(i);
   atr_avg /= 20.0;
   if(atr_avg > 0.0 && atr > atr_avg * InpATRSpikeMult)
   {
      Print("FILTER_REJECT: ATR_SPIKE atr=", DoubleToString(atr, 2),
            " avg=", DoubleToString(atr_avg, 2), " mult=", DoubleToString(InpATRSpikeMult, 2));
      return false;
   }

   return true;
}

int CountOpenPositions()
{
   int count = 0;
   int total = PositionsTotal();
   
   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      
      if(ticket > 0)
      {
         if(PositionSelectByTicket(ticket))
         {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol)
            {
               count++;
            }
         }
      }
   }
   return count;
}

double CalcOpenRiskPercent()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(balance <= 0.0) return 0.0;
   
   double total_risk_usd = 0.0;
   int total = PositionsTotal();
   
   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol)
         {
            double entry  = PositionGetDouble(POSITION_PRICE_OPEN);
            double sl     = PositionGetDouble(POSITION_SL);
            double volume = PositionGetDouble(POSITION_VOLUME);
            
            if(sl > 0.0)
            {
               double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
               double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
               
               // محاسبه ریسک دلاری پوزیشن بر اساس فاصله قیمت ورود تا استاپ لاس
               double risk_dist = MathAbs(entry - sl);
               double position_risk_usd = (risk_dist / tick_size) * tick_value * volume;
               
               total_risk_usd += position_risk_usd;
            }
         }
      }
   }
   
   return (total_risk_usd / balance) * 100.0;
}

int OnInit()
{
   g_atr_handle = iATR(_Symbol, InpSignalTimeframe, InpATRPeriod);
   g_adx_handle = iADX(_Symbol, InpSignalTimeframe, InpADXPeriod);
   if(g_atr_handle == INVALID_HANDLE || g_adx_handle == INVALID_HANDLE)
      return INIT_FAILED;
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_atr_handle);
   if(g_adx_handle != INVALID_HANDLE)
      IndicatorRelease(g_adx_handle);
}

void OnTick()
{
   g_trader.ResetStateIfClosed();
   g_trader.ManagePosition();

   if(!IsNewCandle())
      return;

   if(!PassFilters())
      return;

   g_signal.Analyze();

   if(CountOpenPositions() >= InpMaxPositions)
   {
      Print("SKIP: max positions reached");
      return;
   }

   if(g_last_trade_bar_time != 0)
   {
      int bars_since = (int)((g_last_bar_time - g_last_trade_bar_time) / PeriodSeconds(InpSignalTimeframe));
      if(bars_since < InpMinCandlesBetweenTrades)
      {
         Print("SKIP: min candles between trades not met bars_since=", (string)bars_since);
         return;
      }
   }

   double daily_risk = CalcOpenRiskPercent();
   if(daily_risk >= InpMaxDailyRiskPercent)
   {
      Print("SKIP: max daily risk reached open_risk=", DoubleToString(daily_risk, 2));
      return;
   }

   if(g_trader.HasPosition())
   {
      Print("SKIP: existing position");
      return;
   }

   SSignal signal = g_signal.GenerateSignal();
   if(!signal.valid)
   {
      Print("NO_SIGNAL: no valid breaker/ifvg/momentum");
      return;
   }

   ENUM_ORDER_TYPE order_type = signal.direction == DIR_BULLISH ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   double entry_price = order_type == ORDER_TYPE_BUY ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                                     : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   entry_price = NormalizeDouble(entry_price, _Digits);

   double sl = 0.0;
   double tp1 = 0.0;
   double tp2 = 0.0;
   g_risk.BuildLevels(signal.direction, entry_price, InpSLPips, InpTP1Pips, InpTP2Pips, sl, tp1, tp2);
   if(!InpTP2Enabled)
      tp2 = 0.0;

   double rr = MathAbs(tp1 - entry_price) / MathAbs(entry_price - sl);
   if(rr < InpMinRiskReward)
   {
      Print("ORDER_REJECT: risk reward below min rr=", DoubleToString(rr, 2),
            " min=", DoubleToString(InpMinRiskReward, 2));
      return;
   }

   double volume = g_risk.CalcVolume(InpSLPips, InpRiskPercent, order_type, entry_price);
   if(volume <= 0.0)
   {
      Print("ORDER_REJECT: volume=0, check risk/contract settings");
      return;
   }

   double required_margin = 0.0;
   OrderCalcMargin(order_type, _Symbol, volume, entry_price, required_margin);
   Print("[TRADE_LOG] Vol: ", DoubleToString(volume, 2),
         " SL: ", DoubleToString(sl, _Digits),
         " TP1: ", DoubleToString(tp1, _Digits),
         " Margin: ", DoubleToString(required_margin, 2));

   if(g_trader.OpenPosition(signal, sl, tp1, tp2, volume))
      g_last_trade_bar_time = g_last_bar_time;
   if(g_trader.HasPosition())
      g_signal.DrawSignal(signal);
}


// محاسبه لات سایز با چک کردن مارجین (جلوگیری از خطای 10019)
double CalculateSmartLot(double sl_dist_price)
{
   double risk_usd = AccountInfoDouble(ACCOUNT_BALANCE) * (InpRiskPercent / 100.0);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double entry_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   if(sl_dist_price <= 0) return 0.01;
   
   double lot = risk_usd / (sl_dist_price / tick_size * tick_value);
   
   // محدود کردن بر اساس حداکثر لات فایل کانفیگ (2.0)
   lot = MathMin(lot, 2.0); 

   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lot = MathMax(min_lot, MathMin(max_lot, lot));
   lot = MathFloor(lot / step) * step;

   double free_margin = AccountInfoDouble(ACCOUNT_FREEMARGIN);
   double required_margin = 0.0;
   if(OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, lot, entry_price, required_margin))
   {
      if(required_margin > free_margin * 0.8)
      {
         double scale = (free_margin * 0.8) / required_margin;
         lot = MathFloor((lot * scale) / step) * step;
      }
   }

   return NormalizeDouble(MathMax(lot, min_lot), 2);
}

// تبدیل پیپ به قیمت مخصوص طلا
double PipsToPrice(double pips)
{
   return pips * 0.1;
}
