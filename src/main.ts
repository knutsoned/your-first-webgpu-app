import "./style.css";

// the code from https://codelabs.developers.google.com/your-first-webgpu-app
// with TypeScript declarations and running commentary added

// 32x32 grid of cells
const GRID_SIZE = 32;

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

    // declare the WGSL shaders
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

            // Add 1 to the position before dividing by the grid size

            // adding a scalar to a vector adds the scalar to each component

            // subtracting 1 at the end shifts to bottom left
            let gridPos = (input.pos + 1) / grid - 1 + cellOffset;

            // a var is actually a var
            var output: VertexOutput;
            output.cell = cell;
            output.pos = vec4f(gridPos, 0, 1); // (X, Y, Z, W)
            return output;
        }

        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
          let c = input.cell / grid;
          return vec4f(c, 1-c.x, 0.9); // (Red, Green, Blue, Alpha) -> Red 90%
        }
      `,
    });

    // define the pipeline
    const cellPipeline: GPURenderPipeline = device.createRenderPipeline({
      label: "Cell pipeline",
      layout: "auto",
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

    // the encoder sends commands to the device
    const encoder: GPUCommandEncoder = device.createCommandEncoder();

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

    // describe a grid
    const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
    const uniformBuffer = device.createBuffer({
      label: "Grid Uniforms",
      size: uniformArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // fill the buffer with the array
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    // declare a bind group
    const bindGroup = device.createBindGroup({
      label: "Cell renderer bind group",
      layout: cellPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: uniformBuffer },
        },
      ],
    });

    // and in the darkness...
    pass.setBindGroup(0, bindGroup);

    // let's do this thing
    pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE); // 6 vertices, 16 cells

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
}

console.log("where's the tylenol");
