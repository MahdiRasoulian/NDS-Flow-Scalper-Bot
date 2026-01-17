#pragma once

class CBarSeries
{
private:
   MqlRates m_rates[];
   int      m_copied;

public:
   CBarSeries() : m_copied(0) {}

   bool Refresh(const string symbol, ENUM_TIMEFRAMES tf, const int bars)
   {
      if(bars <= 0)
         return false;
      ArraySetAsSeries(m_rates, true);
      m_copied = CopyRates(symbol, tf, 0, bars, m_rates);
      return (m_copied > 0);
   }

   int Count() const
   {
      return m_copied;
   }

   const MqlRates &Get(const int index) const
   {
      return m_rates[index];
   }
};
