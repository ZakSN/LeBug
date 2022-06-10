`define SIMULATION
`include "delta_compressor.sv"
`include "trace_buffer.sv"
`include "reconfig_unit.sv"

`timescale 1 ns/10 ps

module testbench;

  // DUT parameters
  parameter N=8;
  parameter DATA_WIDTH=8;
  parameter DELTA_SLOTS=2;
  parameter TB_SIZE=8;
  parameter COMPRESSED=0;

  // setup the clock
  localparam period = 10;
  localparam half_period = 5;
  reg clk=1'b0;
  always #half_period clk=~clk;

  // signals
  wire tracing;
  reg [DATA_WIDTH-1:0] vector_in [N-1:0];
  reg valid_in;
  wire compression_flag;
  wire [DATA_WIDTH-1:0] last_vector_out_dc [N-1:0];
  wire [$clog2(TB_SIZE)-1:0] tb_ptr_out;

  wire valid_out;
  wire [DATA_WIDTH-1:0] vector_out [N-1:0];
  wire v_out_comp;
  wire inc_tb_ptr;
  wire compression_flag_out_tb;

  reg new_rx_data;
  reg [7:0] rx_data;
  wire new_tx_data;
  wire [7:0] tx_data;
  reg tx_busy;
  wire [$clog2(TB_SIZE)-1:0] tb_read_address;
  wire [DATA_WIDTH-1:0] vector_out_tb [N-1:0];

  // instantiate DUT
  deltaCompressor #(
    .N(N),
    .DATA_WIDTH(DATA_WIDTH),
    .DELTA_SLOTS(DELTA_SLOTS),
    .COMPRESSED(COMPRESSED)
  )
  dc(
    .clk(clk),
    .tracing(tracing),
    .valid_in(valid_in),
    .vector_in(vector_in),
    .valid_out(valid_out),
    .vector_out(vector_out),
    .inc_tb_ptr(inc_tb_ptr),
    .last_vector_out(last_vector_out_dc),
    .compression_flag_out(compression_flag)
  );
 
 traceBuffer #(
  .N(N),
  .DATA_WIDTH(DATA_WIDTH),
  .TB_SIZE(TB_SIZE)
  )
  tb (
  .clk(clk),
  .tracing(tracing),
  .valid_in(valid_out),
  .compression_flag_in(compression_flag),
  .inc_tb_ptr(inc_tb_ptr),
  .tb_read_address(tb_read_address),
  .vector_in(vector_out),
  .vector_out(vector_out_tb),
  .compression_flag_out(compression_flag_out_tb),
  .tb_ptr_out(tb_ptr_out)
 );

 reconfigUnit #(
  .N(N),
  .M(N),
  .TB_SIZE(TB_SIZE),
  .DATA_WIDTH(DATA_WIDTH),
  .MAX_CHAINS(4),
  .FUVRF_SIZE(4),
  .COMPRESSED(COMPRESSED)
 )
 ru (
  .clk(clk),
  .rx_data(rx_data),
  .new_rx_data(new_rx_data),
  .tx_data(tx_data),
  .new_tx_data(new_tx_data),
  .tx_busy(tx_busy),
  .tracing(tracing),
  .configId(/**/),
  .configData(/**/),
  .tb_mem_address(tb_read_address),
  .vector_out_tb(vector_out_tb),
  .compression_flag_out_tb(compression_flag_out_tb),
  .last_vector_out_dc(last_vector_out_dc),
  .tb_ptr_out_tb(tb_ptr_out)
 );

  integer i;
  always @(posedge clk) begin
    // XXX: simulation only
    #2
    if (inc_tb_ptr == 1'b1) begin
      for (i = 0; i < N; i++) begin
        $write(vector_out[(N-1) - i]);
        $write(", ");
      end
      $display("\n");
    end
  end
  // drive stimulus
  initial begin
    new_rx_data = 1'b0;
    tx_busy = 1'b0;
    //tracing = 1'b0;
    valid_in = 1'b0;
    #half_period;
    #half_period;

    //tracing = 1'b1;
    valid_in = 1'b0;
    #half_period;
    #half_period;

    valid_in = 1'b1;
    vector_in[7:0] = '{1, 2, 3, 4, 5, 6, 7, 8};
    #half_period;
    #half_period; 
    
    valid_in = 1'b1;
    vector_in[7:0] = '{2, 3, 4, 5, 6, 7, 8, 9};
    #half_period;
    #half_period;
    
    valid_in = 1'b1;
    vector_in[7:0] = '{3, 4, 5, 6, 7, 8, 9, 10};
    #half_period;
    #half_period;
    
    valid_in = 1'b1;
    vector_in[7:0] = '{4, 5, 6, 7, 8, 9, 10, 110000};
    #half_period;
    #half_period;
    
    valid_in = 1'b1;
    vector_in[7:0] = '{5, 6, 7, 8, 9, 10, 11, 12};
    #half_period;
    #half_period;
    
    valid_in = 1'b1;
    vector_in[7:0] = '{6, 7, 8, 9, 10, 11, 12, 13};
    #half_period;
    #half_period;
    
    valid_in = 1'b1;
    vector_in[7:0] = '{7, 8, 9, 10, 11, 12, 13, 14};
    #half_period;
    #half_period;
    
    valid_in = 1'b1;
    vector_in[7:0] = '{8, 9, 10, 11, 12, 13, 14, 15};
    #half_period;
    #half_period;
    
    valid_in = 1'b1;
    vector_in[7:0] = '{1, 2, 3, 4, 5, 6, 7, 8};
    #half_period;
    #half_period; 
    
    valid_in = 1'b1;
    vector_in[7:0] = '{2, 3, 4, 5, 6, 7, 8, 9};
    #half_period;
    #half_period;
    
    valid_in = 1'b1;
    vector_in[7:0] = '{3, 4, 5, 6, 7, 8, 9, 10};
    #half_period;
    #half_period;
    
    valid_in = 1'b1;
    vector_in[7:0] = '{4, 5, 6, 7, 8, 9, 10, 110000};
    #half_period;
    #half_period;
    
    valid_in = 1'b1;
    vector_in[7:0] = '{5, 6, 7, 8, 9, 10, 11, 110001};
    #half_period;
    #half_period;
    
    valid_in = 1'b1;
    vector_in[7:0] = '{6, 7, 8, 9, 10, 11, 12, 110002};
    #half_period;
    #half_period;
    
    valid_in = 1'b1;
    vector_in[7:0] = '{7, 8, 9, 10, 11, 12, 13, 14};
    #half_period;
    #half_period;
    
    valid_in = 1'b1;
    vector_in[7:0] = '{15, 9, 15, 11, 15, 13, 15, 15};
    #half_period;
    #half_period;

    valid_in = 1'b0;
    new_rx_data = 1'b1;
    rx_data = 8'b0;
    #half_period;
    #half_period;
   
    while(ru.dbg_state != ru.DBG_TRACING) begin
      #half_period;
      #half_period;
    end

    $finish;
  end

endmodule
