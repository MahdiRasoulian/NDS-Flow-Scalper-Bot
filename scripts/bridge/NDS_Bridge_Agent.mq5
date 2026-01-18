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

struct BridgeRequest
{
   uint magic;
   ushort version;
   ushort flags;
   long sequence;
   long timestamp_ms;
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
   long bar_time_current;
   long bar_time_previous;
   uint market_state;
};

struct BridgeResponse
{
   uint magic;
   ushort version;
   ushort action;
   long sequence;
   double entry;
   double sl;
   double tp;
   double volume;
   double confidence;
   uint flags;
   char json_payload[256];
};

#import "kernel32.dll"
long CreateFileMappingW(int hFile,int lpAttributes,int flProtect,int dwMaximumSizeHigh,int dwMaximumSizeLow,string lpName);
long OpenFileMappingW(int dwDesiredAccess,bool bInheritHandle,string lpName);
long MapViewOfFile(long hFileMappingObject,int dwDesiredAccess,int dwFileOffsetHigh,int dwFileOffsetLow,int dwNumberOfBytesToMap);
bool UnmapViewOfFile(long lpBaseAddress);
bool CloseHandle(long hObject);
long CreateEventW(int lpEventAttributes,bool bManualReset,bool bInitialState,string lpName);
bool SetEvent(long hEvent);
bool ResetEvent(long hEvent);
int WaitForSingleObject(long hHandle,int dwMilliseconds);
void RtlMoveMemory(long dst, uchar &src[], int size);
void RtlMoveMemory(uchar &dst[], long src, int size);
#import

CTrade trade;

const int SYMBOL_SIZE = 12;
const int JSON_SIZE = 256;
const int REQUEST_SIZE = 120;
const int RESPONSE_SIZE = 316;

long mapping_handle = 0;
long map_view = 0;
long request_event = 0;
long response_event = 0;
long sequence_id = 0;
int csv_handle = INVALID_HANDLE;
string resolved_symbol = "";

void WriteUShort(uchar &buffer[], int offset, ushort value)
{
   ShortToCharArray((short)value, buffer, offset);
}

void WriteUInt(uchar &buffer[], int offset, uint value)
{
   IntToCharArray((int)value, buffer, offset);
}

void WriteLong(uchar &buffer[], int offset, long value)
{
   LongToCharArray(value, buffer, offset);
}

void WriteDouble(uchar &buffer[], int offset, double value)
{
   DoubleToCharArray(value, buffer, offset);
}

ushort ReadUShort(const uchar &buffer[], int offset)
{
   return (ushort)CharArrayToShort(buffer, offset);
}

uint ReadUInt(const uchar &buffer[], int offset)
{
   return (uint)CharArrayToInteger(buffer, offset);
}

long ReadLong(const uchar &buffer[], int offset)
{
   return CharArrayToLong(buffer, offset);
}

double ReadDouble(const uchar &buffer[], int offset)
{
   return CharArrayToDouble(buffer, offset);
}

void CopyCharsToBuffer(uchar &buffer[], int offset, const char &source[], int length)
{
   for(int i = 0; i < length; i++)
      buffer[offset + i] = (uchar)source[i];
}

void CopyBufferToChars(const uchar &buffer[], int offset, char &target[], int length)
{
   for(int i = 0; i < length; i++)
      target[i] = (char)buffer[offset + i];
}

void PackRequest(const BridgeRequest &request, uchar &buffer[])
{
   ArrayResize(buffer, REQUEST_SIZE);
   ArrayInitialize(buffer, 0);

   int offset = 0;
   WriteUInt(buffer, offset, request.magic);
   offset += 4;
   WriteUShort(buffer, offset, request.version);
   offset += 2;
   WriteUShort(buffer, offset, request.flags);
   offset += 2;
   WriteLong(buffer, offset, request.sequence);
   offset += 8;
   WriteLong(buffer, offset, request.timestamp_ms);
   offset += 8;
   CopyCharsToBuffer(buffer, offset, request.symbol, SYMBOL_SIZE);
   offset += SYMBOL_SIZE;
   WriteDouble(buffer, offset, request.bid);
   offset += 8;
   WriteDouble(buffer, offset, request.ask);
   offset += 8;
   WriteDouble(buffer, offset, request.spread);
   offset += 8;
   WriteDouble(buffer, offset, request.o_cur);
   offset += 8;
   WriteDouble(buffer, offset, request.h_cur);
   offset += 8;
   WriteDouble(buffer, offset, request.l_cur);
   offset += 8;
   WriteDouble(buffer, offset, request.c_cur);
   offset += 8;
   WriteDouble(buffer, offset, request.o_prev);
   offset += 8;
   WriteDouble(buffer, offset, request.h_prev);
   offset += 8;
   WriteDouble(buffer, offset, request.l_prev);
   offset += 8;
   WriteDouble(buffer, offset, request.c_prev);
   offset += 8;
   WriteLong(buffer, offset, request.bar_time_current);
   offset += 8;
   WriteLong(buffer, offset, request.bar_time_previous);
   offset += 8;
   WriteUInt(buffer, offset, request.market_state);
}

bool UnpackResponse(const uchar &buffer[], BridgeResponse &response)
{
   if(ArraySize(buffer) < RESPONSE_SIZE)
      return false;

   int offset = 0;
   response.magic = ReadUInt(buffer, offset);
   offset += 4;
   response.version = ReadUShort(buffer, offset);
   offset += 2;
   response.action = ReadUShort(buffer, offset);
   offset += 2;
   response.sequence = ReadLong(buffer, offset);
   offset += 8;
   response.entry = ReadDouble(buffer, offset);
   offset += 8;
   response.sl = ReadDouble(buffer, offset);
   offset += 8;
   response.tp = ReadDouble(buffer, offset);
   offset += 8;
   response.volume = ReadDouble(buffer, offset);
   offset += 8;
   response.confidence = ReadDouble(buffer, offset);
   offset += 8;
   response.flags = ReadUInt(buffer, offset);
   offset += 4;
   CopyBufferToChars(buffer, offset, response.json_payload, JSON_SIZE);
   return true;
}

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
   request.sequence = sequence_id++;
   request.timestamp_ms = (long)(TimeCurrent() * 1000);

   StringToCharArray(resolved_symbol, request.symbol, 0, SYMBOL_SIZE - 1);

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

   request.bar_time_current = (long)iTime(resolved_symbol, BarTimeframe, 0);
   request.bar_time_previous = (long)iTime(resolved_symbol, BarTimeframe, 1);

   request.market_state = (uint)(TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) ? 1 : 0);

   uchar buffer[];
   PackRequest(request, buffer);
   RtlMoveMemory(map_view, buffer, REQUEST_SIZE);
   SetEvent(request_event);

   int wait_result = WaitForSingleObject(response_event, DecisionTimeoutMs);
   if(wait_result == WAIT_TIMEOUT)
   {
      Print("⚠️ Bridge response timeout for sequence ", request.sequence);
      return;
   }

   uchar response_buffer[];
   ArrayResize(response_buffer, RESPONSE_SIZE);
   RtlMoveMemory(response_buffer, map_view + (long)REQUEST_SIZE, RESPONSE_SIZE);

   BridgeResponse response;
   ZeroMemory(response);
   if(!UnpackResponse(response_buffer, response))
   {
      Print("⚠️ Invalid response payload size.");
      return;
   }

   if(response.magic != request.magic || response.version != request.version)
   {
      Print("⚠️ Response header mismatch.");
      return;
   }

   if(response.sequence != request.sequence)
   {
      Print("⚠️ Response sequence mismatch.");
      return;
   }

   ExecuteCommand(response);
}

void ExecuteCommand(const BridgeResponse &response)
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
