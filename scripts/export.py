#!/usr/bin/env python

import os
import argparse
import torch
import numpy as np
import pytorch_lightning as pl

from src.models.model import ActuatorModel


def export_model(
    checkpoint_path: str,
    output_dir: str = "models/exported",
    sample_input_size: int = None,
    export_torchscript: bool = True,
    export_onnx: bool = True,
    export_torch: bool = True,
):
    """
    Export trained model to deployment formats.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        output_dir: Directory for saving exported models
        sample_input_size: Size of sample input feature vector
        export_torchscript: Whether to export to TorchScript
        export_onnx: Whether to export to ONNX
        export_torch: Whether to export to PyTorch
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model from checkpoint
    model = ActuatorModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Get input size from model if not provided
    if sample_input_size is None:
        sample_input_size = model.hparams.input_dim
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Input feature size: {sample_input_size}")
    
    # Create sample input (batch size of 1 for inference)
    sample_input = torch.randn(1, sample_input_size)
    
    # Export to TorchScript format
    if export_torchscript:
        script_path = os.path.join(output_dir, "actuator_model.pt")
        try:
            scripted_model = torch.jit.trace(model, sample_input)
            torch.jit.save(scripted_model, script_path)
            print(f"TorchScript model saved to: {script_path}")
            
            # Verify TorchScript model
            loaded_model = torch.jit.load(script_path)
            with torch.no_grad():
                torchscript_output = loaded_model(sample_input)
                original_output = model(sample_input)
                
            # Check if outputs match
            if torch.allclose(torchscript_output, original_output):
                print("✓ TorchScript model verified successfully")
            else:
                print("⚠ TorchScript model verification failed - outputs don't match")
                
        except Exception as e:
            print(f"Error exporting to TorchScript: {e}")
    
    # Export to ONNX format
    if export_onnx:
        onnx_path = os.path.join(output_dir, "actuator_model.onnx")
        try:
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                }
            )
            print(f"ONNX model saved to: {onnx_path}")
            
            # Verify ONNX model if onnx package is available
            try:
                import onnx
                import onnxruntime
                
                # Check ONNX model
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                
                # Run inference with ONNX Runtime
                ort_session = onnxruntime.InferenceSession(onnx_path)
                ort_inputs = {ort_session.get_inputs()[0].name: sample_input.numpy()}
                ort_outputs = ort_session.run(None, ort_inputs)
                
                # Compare with PyTorch model output
                pytorch_output = model(sample_input).detach().numpy()
                
                # Check if outputs match
                if np.allclose(ort_outputs[0], pytorch_output, rtol=1e-3, atol=1e-5):
                    print("✓ ONNX model verified successfully")
                else:
                    print("⚠ ONNX model verification failed - outputs don't match")
                    print(f"  PyTorch output: {pytorch_output}")
                    print(f"  ONNX output: {ort_outputs[0]}")
                
            except ImportError:
                print("⚠ ONNX verification skipped - onnx or onnxruntime package not found")
            except Exception as e:
                print(f"⚠ ONNX verification failed: {e}")
                
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
    
    # Export to PyTorch format (state_dict)
    if export_torch:
        torch_path = os.path.join(output_dir, "actuator_model.pth")
        try:
            torch.save(model.state_dict(), torch_path)
            print(f"PyTorch model state_dict saved to: {torch_path}")
            
            # Save model configuration as well
            config_path = os.path.join(output_dir, "model_config.pt")
            torch.save(model.hparams, config_path)
            print(f"Model configuration saved to: {config_path}")
            
        except Exception as e:
            print(f"Error exporting to PyTorch: {e}")
    
    print("\nExport complete!")
    return {
        "torchscript_path": os.path.join(output_dir, "actuator_model.pt") if export_torchscript else None,
        "onnx_path": os.path.join(output_dir, "actuator_model.onnx") if export_onnx else None,
        "torch_path": os.path.join(output_dir, "actuator_model.pth") if export_torch else None,
    }


def test_inference_speed(
    model_path: str,
    model_type: str = "torchscript",
    num_iterations: int = 1000,
    input_size: int = None,
    batch_size: int = 1,
):
    """
    Test inference speed of exported models.
    
    Args:
        model_path: Path to the exported model
        model_type: Type of model ('torchscript', 'onnx', 'torch')
        num_iterations: Number of inference iterations for benchmarking
        input_size: Size of input feature vector
        batch_size: Batch size for inference
    """
    if model_type == "torchscript":
        # Load TorchScript model
        model = torch.jit.load(model_path)
        
        # Create random input
        if input_size is None:
            # Try to infer from model
            input_size = model.hparams.input_dim if hasattr(model, 'hparams') else 6
        
        x = torch.randn(batch_size, input_size)
        
        # Warm-up
        for _ in range(10):
            _ = model(x)
        
        # Time inference
        import time
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(x)
        
        end_time = time.time()
        
    elif model_type == "onnx":
        try:
            import onnxruntime
            
            # Load ONNX model
            session = onnxruntime.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            # Create random input
            if input_size is None:
                # Try to infer from model
                input_size = session.get_inputs()[0].shape[1]
            
            x = torch.randn(batch_size, input_size).numpy()
            
            # Warm-up
            for _ in range(10):
                _ = session.run(None, {input_name: x})
            
            # Time inference
            import time
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = session.run(None, {input_name: x})
            
            end_time = time.time()
            
        except ImportError:
            print("onnxruntime package not found. Please install with 'pip install onnxruntime'")
            return
        
    elif model_type == "torch":
        # Load PyTorch model
        config_path = os.path.join(os.path.dirname(model_path), "model_config.pt")
        
        if os.path.exists(config_path):
            config = torch.load(config_path)
            model = ActuatorModel(**config)
            model.load_state_dict(torch.load(model_path))
        else:
            print("Model config not found. Using default configuration.")
            # Create model with default config
            model = ActuatorModel(input_dim=6)
            model.load_state_dict(torch.load(model_path))
        
        model.eval()
        
        # Create random input
        if input_size is None:
            # Try to infer from model
            input_size = model.hparams.input_dim
        
        x = torch.randn(batch_size, input_size)
        
        # Warm-up
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        
        # Time inference
        import time
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(x)
        
        end_time = time.time()
        
    else:
        print(f"Unknown model type: {model_type}")
        return
    
    # Calculate statistics
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    inferences_per_second = num_iterations / total_time
    
    print(f"\nInference speed test results ({model_type}):")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Average inference time: {avg_time * 1000:.4f} ms")
    print(f"  Inferences per second: {inferences_per_second:.2f}")
    
    return {
        "model_type": model_type,
        "total_time": total_time,
        "avg_time_ms": avg_time * 1000,
        "inferences_per_second": inferences_per_second
    }


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Export actuator model")
    parser.add_argument(
        "checkpoint_path", type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/exported",
        help="Directory for exported models"
    )
    parser.add_argument(
        "--input_size", type=int, default=None,
        help="Size of input feature vector (inferred from model if not provided)"
    )
    parser.add_argument(
        "--test_speed", action="store_true",
        help="Run inference speed test after export"
    )
    parser.add_argument(
        "--no_torchscript", action="store_true",
        help="Skip TorchScript export"
    )
    parser.add_argument(
        "--no_onnx", action="store_true",
        help="Skip ONNX export"
    )
    parser.add_argument(
        "--no_torch", action="store_true",
        help="Skip PyTorch state_dict export"
    )
    
    args = parser.parse_args()
    
    # Export the model
    export_results = export_model(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        sample_input_size=args.input_size,
        export_torchscript=not args.no_torchscript,
        export_onnx=not args.no_onnx,
        export_torch=not args.no_torch,
    )
    
    # Run speed tests if requested
    if args.test_speed:
        print("\nRunning inference speed tests...")
        
        if export_results["torchscript_path"]:
            test_inference_speed(
                export_results["torchscript_path"],
                model_type="torchscript",
                input_size=args.input_size
            )
        
        if export_results["onnx_path"]:
            test_inference_speed(
                export_results["onnx_path"],
                model_type="onnx",
                input_size=args.input_size
            )
        
        if export_results["torch_path"]:
            test_inference_speed(
                export_results["torch_path"],
                model_type="torch",
                input_size=args.input_size
            ) 