"use client";
import { Camera, CloudUpload, Upload } from "lucide-react";
import { Sheet, SheetClose, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "../sheet";
import { Button } from "../button";
import { Input } from "../input";
import React, { useRef, useState } from "react";

const UploadZone = () => {
  const [image, setImage] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [showPrompt, setShowPrompt] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target?.result as string);
        setShowPrompt(true);
      };
      reader.readAsDataURL(file);
    }
  };

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  };

  const captureImage = () => {
    if (videoRef.current) {
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const imageUrl = canvas.toDataURL("image/png");
        setImage(imageUrl);
        setShowPrompt(true);
        stopCamera();
      }
    }
  };

  const handleSubmit = () => {
    console.log({ image, prompt });
    setImage(null);
    setPrompt("");
    setShowPrompt(false);
  };

  return (
    <Sheet onOpenChange={(open) => !open && stopCamera()}>
      <SheetTrigger asChild>
        <div className="relative rounded-2xl overflow-hidden shadow-xl shadow-emerald-500/25 bg-gradient-to-br from-emerald-500/10 to-emerald-700/10 cursor-pointer">
          <div className="border-2 flex items-center justify-center flex-col border-dashed border-emerald-500/80 rounded-[15px] w-full aspect-video hover:bg-emerald-50/50 transition-colors">
            <CloudUpload strokeWidth={1.6} className="w-14 h-14 text-emerald-500" />
            <p className="text-[16px] text-emerald-500">Upload Crop Photo</p>
          </div>
          <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur-md px-3 py-2 rounded-xl shadow-lg text-[12px] font-bold text-emerald-900 border border-emerald-500/20">
            AI-Powered Analysis
          </div>
        </div>
      </SheetTrigger>
      <SheetContent side="bottom" className="rounded-t-[20px] max-h-[90vh] overflow-y-auto">
        <SheetHeader className="text-center mt-6">
          <SheetTitle className="text-3xl text-emerald-500 font-bold">Crop Health Analysis</SheetTitle>
        </SheetHeader>

        <div className="space-y-6 p-6">
          {!showPrompt ? (
            <div className="space-y-4">
              <div className="flex flex-col sm:flex-row gap-4">
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="flex-1 border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50 transition-colors"
                >
                  <Upload className="w-8 h-8 text-emerald-500 mb-2" />
                  <p className="text-center">Upload Crop Photo</p>
                  <p className="text-sm text-gray-500 text-center">Take a clear photo of affected crop leaves</p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                </div>

                <div
                  onClick={stream ? stopCamera : startCamera}
                  className="flex-1 border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50 transition-colors"
                >
                  <Camera className="w-8 h-8 text-emerald-500 mb-2" />
                  <p className="text-center">{stream ? "Stop Camera" : "Use Camera"}</p>
                  <p className="text-sm text-gray-500 text-center">Capture clear image of crop leaves</p>
                </div>
              </div>

              {stream && (
                <div className="mt-4 space-y-2">
                  <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <Button
                    onClick={captureImage}
                    className="w-full bg-emerald-600 hover:bg-emerald-700"
                  >
                    Capture Image
                  </Button>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-4 flex gap-6">
              {image && (
                <div className="relative bg-gray-100 rounded-lg overflow-hidden h-[500px] flex-1 ">
                  <img src={image} alt="Preview" className="h-full mx-auto object-contain" />
                </div>
              )}
              <div className="flex-1 bg-white p-4 ">
                <div className="flex flex-col space-y-4">
                  <label className="text-lg font-medium">
                    Describe the issue or ask about your crop
                  </label>
                  <textarea
                    placeholder="E.g., Why are my leaves turning yellow? Is this a disease?"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded-md resize-none focus:border-emerald-500 focus:ring-emerald-500"
                    rows={4}
                  />
                </div>
                <div className="flex justify-between pt-6">
                  <Button
                    variant="outline"
                    onClick={() => {
                      setImage(null);
                      setShowPrompt(false);
                    }}
                  >
                    Back
                  </Button>
                  <Button
                    onClick={handleSubmit}
                    disabled={!prompt.trim()}
                    className={`bg-emerald-600 hover:bg-emerald-700 
                      ${!prompt.trim() ? "opacity-50 cursor-not-allowed" : ""}`}
                  >
                    Analyze Crop
                  </Button>
                </div>
              </div>
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
};

export default UploadZone;
