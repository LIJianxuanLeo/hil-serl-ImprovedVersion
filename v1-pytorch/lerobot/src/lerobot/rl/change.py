dataset = None
    if cfg.mode == "record":
        # 1. 获取动作特征名称（针对 gym_hil 的特殊处理）
        if teleop_device is not None:
            action_names = teleop_device.action_features
        else:
            # GymHIL 默认动作：x, y, z, gripper
            action_names = ["x", "y", "z", "gripper"]
            
        # 2. 【关键修复】手动构建正确的 features 字典
        # 每一个特征必须是一个 dict，包含 dtype 和 shape
        features = {
            ACTION: {
                "dtype": "float32", 
                "shape": (len(action_names),), 
                "names": action_names
            },
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }

        # 3. 处理观测值 (Pixels 和 State)
        for key, value in transition[TransitionKey.OBSERVATION].items():
            # 获取实际数据的 shape (去掉 batch 维度)
            val_shape = value.squeeze(0).shape
            
            if "image" in key or "pixels" in key:
                features[key] = {
                    "dtype": "video", # 图像必须指定为 video 才能使用 use_videos=True
                    "shape": val_shape,
                    "names": ["channels", "height", "width"],
                }
            else:
                features[key] = {
                    "dtype": "float32",
                    "shape": val_shape,
                    "names": None,
                }

        # 4. 创建数据集（删除 try-except，让报错直接显现，方便排查）
        import shutil
        from pathlib import Path
        if cfg.dataset.root:
            root_path = Path(cfg.dataset.root)
            if root_path.exists():
                shutil.rmtree(root_path) # 自动清理旧目录防止报错

        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.root,
            use_videos=True,
            image_writer_threads=4,
            features=features,
        )
        print(f"Dataset created successfully at {cfg.dataset.root}")