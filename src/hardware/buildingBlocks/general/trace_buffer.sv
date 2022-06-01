 //-----------------------------------------------------
 // Design Name : Trace Buffer
 // Function    : Circular trace buffer of depth TB_SIZE
 //-----------------------------------------------------
`include "ram_dual_port.sv"

 module  traceBuffer #(
  parameter N=8,
  parameter DATA_WIDTH=32,
  parameter TB_SIZE=64
  )
  (
  input logic clk,
  input logic tracing,
  input logic valid_in,
  input logic compression_flag_in,
  input logic inc_tb_ptr,
  input logic [$clog2(TB_SIZE)-1:0] tb_read_address,
  input logic [DATA_WIDTH-1:0] vector_in [N-1:0],
  output reg [DATA_WIDTH-1:0] vector_out [N-1:0],
  output reg compression_flag_out
 );

    //----------Internal Variables------------
    wire empty,full;

    parameter LATENCY = 2;
    parameter RAM_LATENCY = LATENCY-1;
    parameter MEM_WIDTH = N*DATA_WIDTH;

    //-------------Code Start-----------------

    // Instantiate memory to implement circular trace buffer
    reg [$clog2(TB_SIZE)-1:0] tb_ptr=0;
    wire write_enable_a;
    reg write_enable_b=0;
    wire [MEM_WIDTH-1:0] tbuffer_in_a;
    reg [MEM_WIDTH-1:0] tbuffer_in_b=0;
    wire [MEM_WIDTH-1:0] tbuffer_out_a;
    wire [MEM_WIDTH-1:0] tbuffer_out_b;
    ram_dual_port tbuffer (
      .clk( clk ),
      .clken( 1'b1 ),
      .address_a( tb_ptr ),
      .address_b( tb_read_address ),
      .wren_a( write_enable_a ),
      .wren_b( write_enable_b ),
      .data_a( tbuffer_in_a ),
      .data_b( tbuffer_in_b ),
      .byteena_a( 1'b1 ),
      .byteena_b( 1'b1 ),
      .q_a( tbuffer_out_a ),
      .q_b( tbuffer_out_b)
    );
    defparam tbuffer.width_a = MEM_WIDTH;
    defparam tbuffer.width_b = MEM_WIDTH;
    defparam tbuffer.widthad_a = $clog2(TB_SIZE);
    defparam tbuffer.widthad_b = $clog2(TB_SIZE);
    defparam tbuffer.width_be_a = 1;
    defparam tbuffer.width_be_b = 1;
    defparam tbuffer.numwords_a = TB_SIZE;
    defparam tbuffer.numwords_b = TB_SIZE;
    defparam tbuffer.latency = RAM_LATENCY;
    defparam tbuffer.init_file = "traceBuffer.mif";

    // Instantiate memory to implement circular compression flag buffer
    reg cfbuffer_in_b=0;
    wire cfbuffer_out_a;
    ram_dual_port cfbuffer (
      .clk( clk ),
      .clken( 1'b1 ),
      .address_a( tb_ptr ),
      .address_b( tb_read_address ),
      .wren_a( write_enable_a ),
      .wren_b( write_enable_b ),
      .data_a( compression_flag_in ),
      .data_b( cfbuffer_in_b ),
      .byteena_a( 1'b1 ),
      .byteena_b( 1'b1 ),
      .q_a( cfbuffer_out_a ),
      .q_b( compression_flag_out )
    );
    defparam cfbuffer.width_a = 1;
    defparam cfbuffer.width_b = 1;
    defparam cfbuffer.widthad_a = $clog2(TB_SIZE);
    defparam cfbuffer.widthad_b = $clog2(TB_SIZE);
    defparam cfbuffer.width_be_a = 1;
    defparam cfbuffer.width_be_b = 1;
    defparam cfbuffer.numwords_a = TB_SIZE;
    defparam cfbuffer.numwords_b = TB_SIZE;
    defparam cfbuffer.latency = RAM_LATENCY;
    //defparam cfbuffer.init_file = "compressionFlagBuffer.mif";

    always @(posedge clk) begin

        // Logic for enqueuing values
        if (tracing==1'b1 & valid_in==1'b1 & inc_tb_ptr==1'b1) begin
            tb_ptr <= tb_ptr<TB_SIZE-1 ? tb_ptr+1'b1 : 0;
        end
    end


    // Directly assign module inputs to port A of memory
    assign tbuffer_in_a = { >> { vector_in }};
    assign tbuffer_write_enable_a = valid_in;
    assign cfbuffer_write_enable_a = valid_in;

    // Module output comes from port b (need to drive it when dumping the content)
    assign vector_out = { >> { tbuffer_out_b }};
 
 endmodule 
