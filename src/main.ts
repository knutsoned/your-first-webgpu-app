import "./style.css";

import { mkAlea } from "@spissvinkel/alea";

const { random } = mkAlea();

// the code from https://codelabs.developers.google.com/your-first-webgpu-app
// with TypeScript declarations and running commentary added

// 32x32 grid of cells
const GRID_SIZE = 32;

// for the render loop
const UPDATE_INTERVAL = 200; // Update every 200ms (5 times/sec)

// for the compute shader
const WORKGROUP_SIZE = 8;

let step = 0; // Track how many simulation steps have been run

const gridCount = GRID_SIZE * GRID_SIZE;

// get the canvas element
const canvas = document.querySelector("canvas");

if (canvas) {
  // Your WebGPU code will begin here! (unless it errors out)
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
  }

  // adapter gets the interface we are using
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
  }

  // device is logical (one device can attach to many canvases)
  const device: GPUDevice = await adapter.requestDevice();

  // get the WebGPU context from the DOM
  const context = canvas.getContext("webgpu");

  if (context) {
    // a context is configured to connect this canvas to this device
    const canvasFormat: GPUTextureFormat =
      navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device: device,
      format: canvasFormat,
      alphaMode: "premultiplied", // https://webgpufundamentals.org/webgpu/lessons/webgpu-transparency.html
    });

    // a square
    // prettier-ignore
    const vertices = new Float32Array([
      // X, Y,
      -0.8, -0.8, // Triangle 1 (Blue)
      0.8, -0.8,
      0.8, 0.8,

      -0.8, -0.8, // Triangle 2 (Red)
      0.8, 0.8,
      -0.8, 0.8,
    ]);

    // declare a GPU buffer for the vertices
    const vertexBuffer: GPUBuffer = device.createBuffer({
      label: "Cell vertices",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    // fill the buffer with the array
    device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/ 0, vertices);

    // map the ES array to vertices in the GPU buffer
    const vertexBufferLayout: GPUVertexBufferLayout = {
      arrayStride: 8,
      attributes: [
        {
          format: "float32x2",
          offset: 0,
          shaderLocation: 0, // Position, see vertex shader
        },
      ],
    };

    // describe a grid
    const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
    const uniformBuffer = device.createBuffer({
      label: "Grid Uniforms",
      size: uniformArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // fill the buffer with the array
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    // declare a cell state array and buffer
    const cellStateArray = new Uint32Array(gridCount);
    const cellStateStorage = [
      device.createBuffer({
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      device.createBuffer({
        label: "Cell State B",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
    ];

    // Mark every 3rd cell of the grid as active
    // Set each cell to a random state, then copy the JavaScript array
    // into the storage buffer.
    for (let i = 0; i < cellStateArray.length; ++i) {
      cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
    }
    device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

    // declare the WGSL shaders for rendering
    // need to declare these before defining how to map params (bind groups)
    const cellShaderModule: GPUShaderModule = device.createShaderModule({
      label: "Cell shader",
      // prettier will not reformat WGSL strings but VS Code indenting works
      // kinda

      // special comment triggers the WGSL Literal highlighter
      code: /* wgsl */ `
        struct VertexInput {
          @location(0) pos: vec2f,
          @builtin(instance_index) instance: u32,
        }

        struct VertexOutput {
          @builtin(position) pos: vec4f,
          @location(0) cell: vec2f,
        }

        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> cellState: array<u32>;

        // generate the triangles for a grid of squares
        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
            // let means const, bad is good, welcome back kotter

            // convert the instance index into a grid position (cast to float)
            let i = f32(input.instance);

            // express cell as position, left-right, bottom-top

            // y position is the integer part of the quotient
            // x position is the remainder
            let cell = vec2f(i % grid.x, floor(i / grid.x)); 
            let cellOffset = cell / grid * 2; // Compute the offset to cell

            // cast the cell on/off state to a float
            let state = f32(cellState[input.instance]);

            // Add 1 to the position before dividing by the grid size
            // multiply position by cell state first
            // this makes all vertex coordinates for the inactive cell the same
            // the GPU then ignores the cell

            // adding a scalar to a vector adds the scalar to each component

            // subtracting 1 at the end shifts to bottom left
            let gridPos = (input.pos * state + 1) / grid - 1 + cellOffset;

            // a var is actually a var
            var output: VertexOutput;
            output.cell = cell;
            output.pos = vec4f(gridPos, 0, 1); // (X, Y, Z, W)
            return output;
        }

        // color each square according to grid position
        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
          let c = input.cell / grid; // red and green proportional to x,y in grid
          // decrease blue as the x coordinate gets bigger
          return vec4f(c, 1-c.x, 0.9); // (Red, Green, Blue, Alpha) -> Red 90%
        }
      `,
    });

    // Create the bind group layout and pipeline layout.
    const bindGroupLayout = device.createBindGroupLayout({
      label: "Cell Bind Group Layout",
      entries: [
        {
          binding: 0,
          visibility:
            GPUShaderStage.VERTEX |
            GPUShaderStage.FRAGMENT |
            GPUShaderStage.COMPUTE,
          buffer: {}, // Grid uniform buffer
        },
        {
          binding: 1,
          visibility:
            GPUShaderStage.VERTEX |
            GPUShaderStage.FRAGMENT |
            GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" }, // Cell state input buffer
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }, // Cell state output buffer
        },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      label: "Cell Pipeline Layout",
      bindGroupLayouts: [bindGroupLayout],
    });

    // define the render pipeline
    const cellPipeline: GPURenderPipeline = device.createRenderPipeline({
      label: "Cell pipeline",
      layout: pipelineLayout,
      vertex: {
        module: cellShaderModule,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout],
      },
      fragment: {
        module: cellShaderModule,
        entryPoint: "fragmentMain",
        targets: [
          {
            format: canvasFormat,
          },
        ],
      },
    });

    // the compute shader actually runs the simulation
    const simulationShaderModule = device.createShaderModule({
      label: "Game of Life simulation shader",
      code: /* wgsl */ `
        @group(0) @binding(0) var<uniform> grid: vec2f;

        @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

        fn cellIndex(cell: vec2u) -> u32 {
          // simple indexing
          // return cell.y * u32(grid.x) + cell.x;

          // wrap around to other edge of the grid
          return (cell.y % u32(grid.y)) * u32(grid.x) +
                 (cell.x % u32(grid.x));
        }

        fn cellActive(x: u32, y: u32) -> u32 {
          return cellStateIn[cellIndex(vec2(x, y))];
        }

        @compute
        @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
          // Determine how many active neighbors this cell has.
          let activeNeighbors = cellActive(cell.x+1, cell.y+1) +
                                cellActive(cell.x+1, cell.y) +
                                cellActive(cell.x+1, cell.y-1) +
                                cellActive(cell.x, cell.y-1) +
                                cellActive(cell.x-1, cell.y-1) +
                                cellActive(cell.x-1, cell.y) +
                                cellActive(cell.x-1, cell.y+1) +
                                cellActive(cell.x, cell.y+1);

          let i = cellIndex(cell.xy);

          // Conway's game of life rules:
          switch activeNeighbors {
            case 2: { // Active cells with 2 neighbors stay active.
              cellStateOut[i] = cellStateIn[i];
            }
            case 3: { // Cells with 3 neighbors become or stay active.
              cellStateOut[i] = 1;
            }
            default: { // Cells with < 2 or > 3 neighbors become inactive.
              cellStateOut[i] = 0;
            }
          }
        }
      `,
    });

    // Create a compute pipeline that updates the game state.
    const simulationPipeline = device.createComputePipeline({
      label: "Simulation pipeline",
      layout: pipelineLayout,
      compute: {
        module: simulationShaderModule,
        entryPoint: "computeMain",
      },
    });

    // declare a bind group
    // since this uses the render pipeline for the layout,
    // the render pipeline has to be declared first
    const bindGroup = device.createBindGroup({
      label: "Cell renderer bind group",
      layout: cellPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: uniformBuffer },
        },
        {
          binding: 1,
          resource: { buffer: cellStateStorage[0] },
        },
        {
          binding: 2,
          resource: { buffer: cellStateStorage[1] },
        },
      ],
    });
    const bindGroups = [
      bindGroup,
      device.createBindGroup({
        label: "Cell renderer bind group",
        layout: cellPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: { buffer: uniformBuffer },
          },
          {
            binding: 1,
            resource: { buffer: cellStateStorage[1] },
          },
          {
            binding: 2,
            resource: { buffer: cellStateStorage[0] },
          },
        ],
      }),
    ];

    // RENDER FUNCTION
    function updateGrid() {
      // the encoder sends commands to the device
      const encoder: GPUCommandEncoder = device.createCommandEncoder();

      // run the compute shader
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(simulationPipeline);
      computePass.setBindGroup(0, bindGroups[step % 2]);

      // divide the amount of work by the workgroup size
      const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);

      // actually create the (n X n matrix of) workgroups
      computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
      computePass.end();

      // a pass is a set of operations. this one clears the buffer
      const pass: GPURenderPassEncoder = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(), // args could define a rect on canvas
            loadOp: "clear", // start blank
            // alpha has no effect here unless alphaMode: "premultiplied" is in ctx config
            clearValue: { r: 0.3, g: 0, b: 0.4, a: 0.2 }, // purple 20% [0.3, 0, 0.4, 0.2]
            storeOp: "store", // save when done
          },
        ],
      });

      // use this pipeline
      pass.setPipeline(cellPipeline);

      // with this buffer
      pass.setVertexBuffer(0, vertexBuffer);

      // and in the darkness...
      pass.setBindGroup(0, bindGroups[++step % 2]);
      // the step must be incremented between compute and render pipelines
      // otherwise the render will be a step behind the compute

      // let's do this thing
      pass.draw(vertices.length / 2, gridCount); // 6 vertices / square

      // that's all for now
      pass.end();

      // go ahead...
      //const commandBuffer = encoder.finish();

      // make my day
      //device.queue.submit([commandBuffer]);

      // all together now
      device.queue.submit([encoder.finish()]);

      console.log("holy shit");
    }

    // now run it
    setInterval(updateGrid, UPDATE_INTERVAL);
  }
}

console.log("where's the tylenol");
