 //-----------------------------------------------------
 // Design Name : Delta Compressor
 // Function    : Performs delta compression on N vectors before sending them to the tracebuffer
 //-----------------------------------------------------
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
  reg [DATA_WIDTH-1:0] comp_reg [N-1:0];
  reg [DATA_WIDTH-1:0] cr_upper_mask;
  reg [DATA_WIDTH-1:0] cr_lower_mask;
  reg first_cycle;
  reg [N-1:0] overflow;

  integer i;
  integer j;
  integer ptr;

  always @(posedge clk) begin
    if (valid_in == 1'b1 && tracing==1'b0) begin
      // starting condition; last_vector is invalid and must be re-initialized
      if (first_cycle == 1'b1) begin
        last_vector = vector_in;
        valid_out = 1'b0;
        first_cycle = 1'b0;
        for (i = 0; i < N; i++) begin
          comp_reg[i] = NODATA;
        end
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
          // update the compression register
          cr_upper_mask = {DATA_WIDTH{1'b1}};
          cr_lower_mask = {DATA_WIDTH{1'b1}};
          for (j = 0; j < DELTA_SLOTS; j++) begin
            if (j < (DELTA_SLOTS-ptr)) begin
              cr_upper_mask = cr_upper_mask << PRECISION;
            end
            if (j <= ptr) begin
              cr_lower_mask = cr_lower_mask >> PRECISION;
            end
          end
          for (i = 0; i < N; i++) begin
            comp_reg[i] = (comp_reg[i]&cr_upper_mask) + ((delta[i]&{PRECISION{1'b1}}) << (DELTA_SLOTS-ptr-1)*(PRECISION)) + (NODATA&cr_lower_mask);
          end

          // set output signals
          valid_out = 1'b1;
          v_out_comp = COMPRESSED;
          vector_out = comp_reg;
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
          ptr = 0;
          for (i = 0; i < N; i++) begin
            comp_reg[i] = NODATA;
          end
        end
        last_vector = vector_in;
      end
    end

    // if we're not processing reset the compression algorithm
    if (tracing == 1'b1) begin
      first_cycle = 1'b1;
      ptr = -1;
      valid_out = 1'b0;
    end
  end

endmodule
