#pragma once

struct RiskPlan
{
   double entry;
   double sl;
   double tp1;
   double tp2;
   double lot;
};

class CRiskManager
{
private:
   double m_point;
   int    m_digits;

public:
   CRiskManager() : m_point(0.0), m_digits(0) {}

   void Init(const string symbol)
   {
      m_point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      m_digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   }

   double PointsPerPip() const
   {
      if(m_digits == 3 || m_digits == 5)
         return 10.0;
      return 1.0;
   }

   RiskPlan BuildPlan(const double entry, const double sl_pips, const double tp1_pips, const double tp2_pips,
                      const double lots)
   {
      double pip_points = PointsPerPip();
      RiskPlan plan;
      plan.entry = entry;
      plan.sl = entry - sl_pips * pip_points * m_point;
      plan.tp1 = entry + tp1_pips * pip_points * m_point;
      plan.tp2 = entry + tp2_pips * pip_points * m_point;
      plan.lot = lots;
      return plan;
   }
};
