 //-----------------------------------------------------
 // Design Name : Delta Compressor
 // Function    : Performs delta compression on N vectors before sending them to the tracebuffer
 //-----------------------------------------------------

module compressionRegisterAtom #(
 parameter PRECISION,
 parameter INV,
 parameter ID
 )
 (
  input integer ptr,
  input logic [PRECISION-1:0] i,
  input logic [PRECISION-1:0] delta,
  output reg [PRECISION-1:0] o
 );

 always_comb begin
    if (ptr < ID) begin
      o <= INV;
    end
    else if (ptr == ID) begin
      o <= delta;
    end
    else begin
      o <= i;
    end
 end
endmodule

module compressionRegisterSlice #(
 parameter DELTA_SLOTS,
 parameter PRECISION,
 parameter INV
 )
 (
  input integer ptr,
  input logic [(DELTA_SLOTS*PRECISION)-1:0] I,
  input logic [(DELTA_SLOTS*PRECISION)-1:0] delta,
  output reg [(DELTA_SLOTS*PRECISION)-1:0] O
 );

  genvar atomID;
  generate
    for ( atomID = 0; atomID < DELTA_SLOTS; atomID++) begin: atomIDgen
      compressionRegisterAtom #(
        .PRECISION(PRECISION),
        .INV(INV),
        .ID(DELTA_SLOTS - 1 - atomID)
      )
      cra (
        .ptr(ptr),
        .i(I[(atomID+1)*PRECISION-1:atomID*PRECISION]),
        .delta(delta[PRECISION-1:0]),
        .o(O[(atomID+1)*PRECISION-1:atomID*PRECISION])
      );
    end
  endgenerate
endmodule

module deltaCompressor #(
 parameter N=8,
 parameter DATA_WIDTH=32,
 parameter DELTA_SLOTS=4,
 parameter COMPRESSED=0
 )
 (
 input logic clk,
 input logic tracing,
 input logic valid_in,
 input logic [DATA_WIDTH-1:0] vector_in [N-1:0],
 output reg valid_out,
 output reg [DATA_WIDTH-1:0] vector_out [N-1:0],
 output reg v_out_comp,
 output reg inc_tb_ptr
 );

  // the number of bits available to represent a delta
  parameter PRECISION = DATA_WIDTH/DELTA_SLOTS;
  // the maximum valid delta (signed DATA_WIDTH bit constant)
  parameter DELTA_MAX = {{(DELTA_SLOTS-1)*PRECISION{1'b0}}, {PRECISION-1{1'b1}} };
  // the minimum valid delta (signed DATA_WIDTH bit constant), most negative
  // delta is reserved for the INV symbol
  parameter DELTA_MIN = {{(DELTA_SLOTS-1)*PRECISION{1'b1}}, {PRECISION-2{1'b0}} , 1'b1};
  // INV symbol - invalid delta
  parameter INV = {1'b1, {PRECISION-1{1'b0}}};
  // nodata symbol - empty compression register
  parameter NODATA = {DELTA_SLOTS{INV}};

  reg [DATA_WIDTH-1:0] last_vector [N-1:0];
  reg [DATA_WIDTH-1:0] delta [N-1:0];
  reg [DATA_WIDTH-1:0] comp_reg_D [N-1:0];
  reg [DATA_WIDTH-1:0] comp_reg_Q [N-1:0];
  reg first_cycle;
  reg [N-1:0] overflow;

  integer i;
  integer ptr;

  genvar CrSlice;
  generate
    for ( CrSlice = 0; CrSlice < N; CrSlice++) begin: CrSliceGen
      compressionRegisterSlice #(
       .DELTA_SLOTS(DELTA_SLOTS),
       .PRECISION(PRECISION),
       .INV(INV)
       )
       crs (
        .ptr(ptr),
        .I(comp_reg_Q[CrSlice]),
        .delta(delta[CrSlice]),
        .O(comp_reg_D[CrSlice])
       );
    end
  endgenerate

  always @(posedge clk) begin
    if (valid_in == 1'b1 && tracing==1'b0) begin
      comp_reg_Q = comp_reg_D;
      // starting condition; last_vector is invalid and must be re-initialized
      if (first_cycle == 1'b1) begin
        last_vector = vector_in;
        valid_out = 1'b0;
        first_cycle = 1'b0;
      end
      else begin
        // we've seen at least one vector; start compressing

        // compute delta vector:
        for (i = 0; i < N; i++) begin
          delta[i] = last_vector[i] - vector_in[i];
        end

        // check for overflow
        for (i = 0; i < N; i++) begin
          overflow[i] = (signed'(delta[i]) > signed'(DELTA_MAX)) || (signed'(delta[i]) < signed'(DELTA_MIN));
        end

        if (|overflow != 1'b1) begin // no overflow
          // set output signals
          valid_out = 1'b1;
          v_out_comp = COMPRESSED;
          vector_out = comp_reg_D;
          if (ptr==0) begin
            inc_tb_ptr = 1'b1;
          end
          else begin
            inc_tb_ptr = 1'b0;
          end
          // update pointer
          if (ptr < DELTA_SLOTS) begin
            ptr++;
          end
          if (ptr == DELTA_SLOTS) begin
            ptr = 0;
          end
        end
        else begin // overflow
          // set output signals
          valid_out = 1'b1;
          v_out_comp = ~COMPRESSED;
          vector_out = last_vector;
          inc_tb_ptr = 1'b1;
          ptr = -1;
        end
        last_vector = vector_in;
      end
    end

    // if we're not processing reset the compression algorithm
    if (tracing == 1'b1) begin
      first_cycle = 1'b1;
      ptr = -1;
      valid_out = 1'b0;
      for (i = 0; i < N; i++) begin
        comp_reg_Q[i] = {DELTA_SLOTS{INV}};
      end
    end
  end

  always @(negedge clk) begin
    // XXX: simulation only
    if (inc_tb_ptr == 1'b1) begin
      for (i = 0; i < N; i++) begin
        $write(vector_out[(N-1) - i]);
        $write(", ");
      end
      $display("\n");
    end
 end

endmodule
