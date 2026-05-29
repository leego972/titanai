import React, { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Sphere, MeshDistortMaterial, Float, PerspectiveCamera, Environment, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';

export type Emotion = 'neutral' | 'smiling' | 'laughing' | 'serious' | 'empathetic' | 'thinking' | 'concerned' | 'amused' | 'friendly_stern';

interface DestroMaskProps {
  volume: number;
  emotion: Emotion;
}

const DestroMask = ({ volume = 0, emotion = 'neutral' }: DestroMaskProps) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const leftEyeRef = useRef<THREE.Mesh>(null);
  const rightEyeRef = useRef<THREE.Mesh>(null);
  const jawRef = useRef<THREE.Mesh>(null);
  const { viewport } = useThree();

  // Responsive scaling for the mask
  const responsiveScale = useMemo(() => {
    const baseScale = viewport.width < 3 ? viewport.width / 3.5 : 1;
    return [baseScale, baseScale * 1.3, baseScale * 0.85] as [number, number, number];
  }, [viewport.width]);

  useFrame((state) => {
    if (!meshRef.current) return;
    const time = state.clock.getElapsedTime();
    
    // Base breathing
    meshRef.current.position.y = Math.sin(time * 0.5) * 0.05;
    
    // Default values
    let eyeColor = '#00ffff';
    let glowIntensity = 3 + volume * 20;
    let jawScaleX = 0.6;
    let distort = 0.15;
    let speed = 1.5;

    switch (emotion) {
      case 'thinking':
        eyeColor = '#00ccff';
        glowIntensity = 2 + Math.sin(time * 3) * 1.5;
        distort = 0.1;
        break;
      case 'concerned':
        eyeColor = '#44ccff';
        glowIntensity = 1.5 + volume * 8;
        jawScaleX = 0.55;
        break;
      case 'amused':
        eyeColor = '#00ffff';
        glowIntensity = 4 + volume * 25;
        jawScaleX = 0.75;
        distort = 0.25;
        break;
      case 'friendly_stern':
        eyeColor = '#00eebb';
        glowIntensity = 3 + volume * 15;
        jawScaleX = 0.68;
        break;
      case 'laughing':
        eyeColor = '#00ffff';
        glowIntensity = 6 + Math.sin(time * 20) * 4;
        jawScaleX = 0.95;
        distort = 0.35;
        speed = 4;
        break;
      case 'smiling':
        jawScaleX = 0.85;
        break;
      case 'serious':
        eyeColor = '#0088ff';
        glowIntensity = 1.5 + volume * 5;
        break;
      case 'empathetic':
        eyeColor = '#00ffaa';
        glowIntensity = 2 + volume * 10;
        break;
    }

    if (leftEyeRef.current && rightEyeRef.current) {
      const matL = leftEyeRef.current.material as THREE.MeshStandardMaterial;
      const matR = rightEyeRef.current.material as THREE.MeshStandardMaterial;
      matL.color.set(eyeColor);
      matL.emissive.set(eyeColor);
      matL.emissiveIntensity = glowIntensity;
      matR.color.set(eyeColor);
      matR.emissive.set(eyeColor);
      matR.emissiveIntensity = glowIntensity;
    }

    if (jawRef.current) {
      const jawTarget = -0.65 - (volume * 0.7);
      jawRef.current.position.y = THREE.MathUtils.lerp(jawRef.current.position.y, jawTarget, 0.2);
      jawRef.current.scale.x = THREE.MathUtils.lerp(jawRef.current.scale.x, jawScaleX, 0.1);
      if (emotion === 'laughing') {
        jawRef.current.position.y -= Math.sin(time * 30) * 0.1;
      }
    }

    const mat = meshRef.current.material as any;
    if (mat.distort !== undefined) {
      mat.distort = THREE.MathUtils.lerp(mat.distort, distort + (volume * 0.5), 0.1);
      mat.speed = speed;
    }
  });

  return (
    <group>
      {/* Main Beryllium Steel Mask */}
      <Sphere ref={meshRef} args={[1, 128, 128]} scale={responsiveScale}>
        <MeshDistortMaterial
          color="#e0e0e0"
          roughness={0.05}
          metalness={1}
          distort={0.15}
          speed={1.5}
          envMapIntensity={2}
        />
      </Sphere>
      
      {/* Glowing Eyes - Also scaled based on viewport if needed, but relative position is better */}
      <group scale={viewport.width < 3 ? viewport.width / 3.5 : 1}>
        <mesh ref={leftEyeRef} position={[-0.38, 0.45, 0.65]}>
          <sphereGeometry args={[0.12, 32, 32]} />
          <meshStandardMaterial color="#00ffff" emissive="#00ffff" emissiveIntensity={5} />
        </mesh>
        
        <mesh ref={rightEyeRef} position={[0.38, 0.45, 0.65]}>
          <sphereGeometry args={[0.12, 32, 32]} />
          <meshStandardMaterial color="#00ffff" emissive="#00ffff" emissiveIntensity={5} />
        </mesh>
        
        {/* Articulated Jaw */}
        <mesh ref={jawRef} position={[0, -0.7, 0.35]} scale={[0.65, 0.45, 0.45]}>
          <sphereGeometry args={[1, 64, 64]} />
          <meshStandardMaterial color="#a0a0a0" metalness={1} roughness={0.1} envMapIntensity={1.5} />
        </mesh>
      </group>
    </group>
  );
};

export const DestroFace = ({ volume = 0, emotion = 'neutral' }: { volume?: number, emotion?: Emotion }) => {
  return (
    <div className="w-full h-full min-h-[250px] sm:min-h-[350px] md:min-h-[450px] bg-[#0a0a0a] rounded-xl overflow-hidden shadow-[0_0_50px_rgba(0,0,0,0.8)] border border-white/5 relative flex items-center justify-center">
      <Canvas shadows dpr={[1, 2]} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}>
        <PerspectiveCamera makeDefault position={[0, 0, 5]} fov={40} />
        
        {/* Studio Lighting Setup */}
        <ambientLight intensity={0.4} />
        <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={2} castShadow />
        <pointLight position={[-10, -10, -10]} intensity={1} color="#0088ff" />
        <pointLight position={[0, 5, 10]} intensity={1.5} color="#ffffff" />
        <pointLight position={[5, -5, 5]} intensity={0.8} color="#00ffff" />
        
        <Environment preset="city" />
        
        <Float speed={1.5} rotationIntensity={0.3} floatIntensity={0.4}>
          <DestroMask volume={volume} emotion={emotion} />
        </Float>
        
        <ContactShadows position={[0, -2.5, 0]} opacity={0.4} scale={10} blur={2.5} far={4} />
      </Canvas>
      
      {/* Subtle overlay gradient for depth */}
      <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-black/40 via-transparent to-transparent" />
    </div>
  );
};
