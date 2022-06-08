`include "debugProcessor.sv"

`timescale 1 ns / 1 ns
module top (
      CLOCK_50,
      KEY,
      UART_RXD,
      UART_TXD
      );

    // Compile-time parameters
    parameter N=16;
    parameter M=4;
    parameter DATA_WIDTH=32;
    parameter IB_DEPTH=32;
    parameter MAX_CHAINS=4;
    parameter TB_SIZE=8;
    parameter FUVRF_SIZE=4;
    parameter VVVRF_SIZE=8;

   input CLOCK_50;
   input [3:0] KEY;
   input  UART_RXD;
   output UART_TXD;
   wire   clk = CLOCK_50;
   wire   reset = ~KEY[0];
   wire   start;
   wire   finish;

    // Declare inputs
    reg valid=1'b1;
    reg [1:0] eof=2'b00;
    reg [DATA_WIDTH-1:0] vector [N-1:0] /*synthesis keep*/;

    integer k=0;
    always @(posedge clk) begin
      for (k=0;k<N;k++) begin
        vector[k]<=vector[k]+k+UART_TXD;
      end
      valid<=valid+1'b1;
      eof<=eof+1;
    end

    // Declare outputs
    reg [DATA_WIDTH-1:0] vector_out [N-1:0] /*synthesis keep*/;
    reg valid_out;

    // Instantiate debugger
    debugger #(
      .N(N),
      .M(M),
      .DATA_WIDTH(DATA_WIDTH),
      .IB_DEPTH(IB_DEPTH),
      .MAX_CHAINS(MAX_CHAINS),
      .TB_SIZE(TB_SIZE),
      .FUVRF_SIZE(FUVRF_SIZE),
      .VVVRF_SIZE(VVVRF_SIZE)
    )
    dbg(
      .clk(clk),
      .vector_in(vector),
      .enqueue(valid),
      .eof_in(eof),
      .reset(reset),
      .uart_rxd(UART_RXD),
      .uart_txd(UART_TXD)
    );

endmodule
