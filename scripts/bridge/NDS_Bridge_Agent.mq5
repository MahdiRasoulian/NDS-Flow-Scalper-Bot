#property strict
#property version   "1.00"
#property description "NDS Flow Scalper - MMF Bridge Agent"

#include <Trade/Trade.mqh>

input string BridgeMappingName = "NDS_FLOW_BRIDGE";
input string BridgeRequestEvent = "NDS_FLOW_BRIDGE_REQ";
input string BridgeResponseEvent = "NDS_FLOW_BRIDGE_RESP";
input string TradingSymbol = "XAUUSD!";
input ENUM_TIMEFRAMES BarTimeframe = PERIOD_M15;
input bool UseBarSync = false;
input int DecisionTimeoutMs = 500;
input int DeviationPoints = 30;
input ulong MagicNumber = 909090;

#define INVALID_HANDLE_VALUE -1
#define FILE_MAP_ALL_ACCESS 0x000f001f
#define PAGE_READWRITE 0x04
#define WAIT_OBJECT_0 0x00000000
#define WAIT_TIMEOUT 0x00000102

#pragma pack(push,1)
struct BridgeRequest
{
   uint magic;
   ushort version;
   ushort flags;
   longlong sequence;
   longlong timestamp_ms;
   char symbol[12];
   double bid;
   double ask;
   double spread;
   double o_cur;
   double h_cur;
   double l_cur;
   double c_cur;
   double o_prev;
   double h_prev;
   double l_prev;
   double c_prev;
   longlong bar_time_current;
   longlong bar_time_previous;
   uint market_state;
};

struct BridgeResponse
{
   uint magic;
   ushort version;
   ushort action;
   longlong sequence;
   double entry;
   double sl;
   double tp;
   double volume;
   double confidence;
   uint flags;
   char json_payload[256];
};
#pragma pack(pop)

#import "kernel32.dll"
int CreateFileMappingW(int hFile,int lpAttributes,int flProtect,int dwMaximumSizeHigh,int dwMaximumSizeLow,string lpName);
int OpenFileMappingW(int dwDesiredAccess,bool bInheritHandle,string lpName);
int MapViewOfFile(int hFileMappingObject,int dwDesiredAccess,int dwFileOffsetHigh,int dwFileOffsetLow,int dwNumberOfBytesToMap);
bool UnmapViewOfFile(int lpBaseAddress);
bool CloseHandle(int hObject);
int CreateEventW(int lpEventAttributes,bool bManualReset,bool bInitialState,string lpName);
bool SetEvent(int hEvent);
bool ResetEvent(int hEvent);
int WaitForSingleObject(int hHandle,int dwMilliseconds);
void RtlMoveMemory(int dst, uchar &src[], int size);
void RtlMoveMemory(uchar &dst[], int src, int size);
#import

CTrade trade;

int mapping_handle = 0;
int map_view = 0;
int request_event = 0;
int response_event = 0;
ulong sequence_id = 0;
int csv_handle = INVALID_HANDLE;
string resolved_symbol = "";

string ResolveTradingSymbol(const string desired)
{
   string candidate = desired;
   if(candidate == "")
      candidate = _Symbol;

   if(SymbolSelect(candidate, true))
      return candidate;

   if(StringFind(_Symbol, candidate) == 0)
   {
      SymbolSelect(_Symbol, true);
      return _Symbol;
   }

   if(StringFind(candidate, _Symbol) == 0)
   {
      SymbolSelect(_Symbol, true);
      return _Symbol;
   }

   int total = SymbolsTotal(true);
   for(int i = 0; i < total; i++)
   {
      string name = SymbolName(i, true);
      if(StringFind(name, candidate) == 0)
      {
         SymbolSelect(name, true);
         return name;
      }
   }

   return candidate;
}

int OnInit()
{
   resolved_symbol = ResolveTradingSymbol(TradingSymbol);
   if(!SymbolSelect(resolved_symbol, true))
   {
      Print("❌ Failed to select trading symbol: ", resolved_symbol);
      return INIT_FAILED;
   }

   mapping_handle = CreateFileMappingW(INVALID_HANDLE_VALUE, 0, PAGE_READWRITE, 0, 2048, BridgeMappingName);
   if(mapping_handle == 0)
   {
      Print("❌ Failed to create file mapping.");
      return INIT_FAILED;
   }

   map_view = MapViewOfFile(mapping_handle, FILE_MAP_ALL_ACCESS, 0, 0, 2048);
   if(map_view == 0)
   {
      Print("❌ Failed to map view of file.");
      CloseHandle(mapping_handle);
      return INIT_FAILED;
   }

   request_event = CreateEventW(0, false, false, BridgeRequestEvent);
   response_event = CreateEventW(0, false, false, BridgeResponseEvent);
   if(request_event == 0 || response_event == 0)
   {
      Print("❌ Failed to create events for bridge.");
      return INIT_FAILED;
   }

   trade.SetExpertMagicNumber((int)MagicNumber);
   trade.SetDeviationInPoints(DeviationPoints);

   csv_handle = FileOpen("nds_bridge_exec_log.csv", FILE_WRITE | FILE_CSV | FILE_COMMON, ';');
   if(csv_handle != INVALID_HANDLE)
   {
      FileWrite(csv_handle, "timestamp", "sequence", "action", "entry", "sl", "tp", "volume", "retcode");
   }

   if(UseBarSync)
   {
      EventSetTimer(1);
   }

   Print("✅ NDS Bridge Agent initialized.");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(UseBarSync)
      EventKillTimer();

   if(csv_handle != INVALID_HANDLE)
      FileClose(csv_handle);

   if(map_view != 0)
      UnmapViewOfFile(map_view);
   if(mapping_handle != 0)
      CloseHandle(mapping_handle);
   if(request_event != 0)
      CloseHandle(request_event);
   if(response_event != 0)
      CloseHandle(response_event);

   Print("🧹 Bridge Agent shutdown.");
}

void OnTick()
{
   if(!UseBarSync)
      ProcessBridge();
}

void OnTimer()
{
   if(UseBarSync)
      ProcessBridge();
}

void ProcessBridge()
{
   if(resolved_symbol == "")
      resolved_symbol = ResolveTradingSymbol(TradingSymbol);

   BridgeRequest request;
   ZeroMemory(request);

   request.magic = 0x4E445342;
   request.version = 1;
   request.flags = 0;
   request.sequence = (longlong)sequence_id++;
   request.timestamp_ms = (longlong)(TimeCurrent() * 1000);

   StringToCharArray(resolved_symbol, request.symbol, 0, 11);

   double bid = SymbolInfoDouble(resolved_symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(resolved_symbol, SYMBOL_ASK);
   request.bid = bid;
   request.ask = ask;
   request.spread = MathAbs(ask - bid);

   request.o_cur = iOpen(resolved_symbol, BarTimeframe, 0);
   request.h_cur = iHigh(resolved_symbol, BarTimeframe, 0);
   request.l_cur = iLow(resolved_symbol, BarTimeframe, 0);
   request.c_cur = iClose(resolved_symbol, BarTimeframe, 0);

   request.o_prev = iOpen(resolved_symbol, BarTimeframe, 1);
   request.h_prev = iHigh(resolved_symbol, BarTimeframe, 1);
   request.l_prev = iLow(resolved_symbol, BarTimeframe, 1);
   request.c_prev = iClose(resolved_symbol, BarTimeframe, 1);

   request.bar_time_current = (longlong)iTime(resolved_symbol, BarTimeframe, 0);
   request.bar_time_previous = (longlong)iTime(resolved_symbol, BarTimeframe, 1);

   request.market_state = (uint)(TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) ? 1 : 0);

   uchar buffer[];
   ArrayResize(buffer, sizeof(request));
   StructToCharArray(request, buffer, 0, sizeof(request));
   RtlMoveMemory(map_view, buffer, sizeof(request));
   SetEvent(request_event);

   int wait_result = WaitForSingleObject(response_event, DecisionTimeoutMs);
   if(wait_result == WAIT_TIMEOUT)
   {
      Print("⚠️ Bridge response timeout for sequence ", request.sequence);
      return;
   }

   uchar response_buffer[];
   ArrayResize(response_buffer, sizeof(BridgeResponse));
   RtlMoveMemory(response_buffer, map_view + sizeof(request), sizeof(BridgeResponse));

   BridgeResponse response;
   CharArrayToStruct(response_buffer, response, 0, sizeof(response));

   if(response.sequence != request.sequence)
   {
      Print("⚠️ Response sequence mismatch.");
      return;
   }

   ExecuteCommand(response);
}

void ExecuteCommand(BridgeResponse response)
{
   if(response.action == 0)
      return;

   bool result = false;
   string action_name = "NONE";

   if(response.action == 1)
   {
      action_name = "BUY";
      result = trade.Buy(response.volume, resolved_symbol, response.entry, response.sl, response.tp, "NDS_Bridge");
   }
   else if(response.action == 2)
   {
      action_name = "SELL";
      result = trade.Sell(response.volume, resolved_symbol, response.entry, response.sl, response.tp, "NDS_Bridge");
   }
   else if(response.action == 3)
   {
      action_name = "BUY_LIMIT";
      result = trade.BuyLimit(response.volume, response.entry, resolved_symbol, response.sl, response.tp, ORDER_TIME_GTC, 0, "NDS_Bridge");
   }
   else if(response.action == 4)
   {
      action_name = "SELL_LIMIT";
      result = trade.SellLimit(response.volume, response.entry, resolved_symbol, response.sl, response.tp, ORDER_TIME_GTC, 0, "NDS_Bridge");
   }
   else if(response.action == 5)
   {
      action_name = "BUY_STOP";
      result = trade.BuyStop(response.volume, response.entry, resolved_symbol, response.sl, response.tp, ORDER_TIME_GTC, 0, "NDS_Bridge");
   }
   else if(response.action == 6)
   {
      action_name = "SELL_STOP";
      result = trade.SellStop(response.volume, response.entry, resolved_symbol, response.sl, response.tp, ORDER_TIME_GTC, 0, "NDS_Bridge");
   }

   ulong retcode = trade.ResultRetcode();
   if(!result)
   {
      PrintFormat("❌ Order failed: %s retcode=%d", action_name, retcode);
   }
   else
   {
      PrintFormat("✅ Order sent: %s volume=%.2f entry=%.2f", action_name, response.volume, response.entry);
   }

   if(csv_handle != INVALID_HANDLE)
   {
      FileWrite(csv_handle,
         TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
         (string)response.sequence,
         action_name,
         DoubleToString(response.entry, _Digits),
         DoubleToString(response.sl, _Digits),
         DoubleToString(response.tp, _Digits),
         DoubleToString(response.volume, 2),
         (string)retcode
      );
      FileFlush(csv_handle);
   }
}
