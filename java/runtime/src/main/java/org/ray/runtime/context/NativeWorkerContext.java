package org.ray.runtime.context;

import java.nio.ByteBuffer;
import org.ray.api.id.ActorId;
import org.ray.api.id.JobId;
import org.ray.api.id.TaskId;
import org.ray.api.id.UniqueId;
import org.ray.runtime.generated.Common.TaskType;

/**
 * Worker context for cluster mode. This is a wrapper class for worker context of core worker.
 */
public class NativeWorkerContext implements WorkerContext {

  private final ThreadLocal<ClassLoader> currentClassLoader = new ThreadLocal<>();

  @Override
  public UniqueId getCurrentWorkerId() {
    return UniqueId.fromByteBuffer(nativeGetCurrentWorkerId());
  }

  @Override
  public JobId getCurrentJobId() {
    return JobId.fromByteBuffer(nativeGetCurrentJobId());
  }

  @Override
  public ActorId getCurrentActorId() {
    return ActorId.fromByteBuffer(nativeGetCurrentActorId());
  }

  @Override
  public ClassLoader getCurrentClassLoader() {
    return currentClassLoader.get();
  }

  @Override
  public void setCurrentClassLoader(ClassLoader currentClassLoader) {
    if (this.currentClassLoader.get() != currentClassLoader) {
      this.currentClassLoader.set(currentClassLoader);
    }
  }

  @Override
  public TaskType getCurrentTaskType() {
    return TaskType.forNumber(nativeGetCurrentTaskType());
  }

  @Override
  public TaskId getCurrentTaskId() {
    return TaskId.fromByteBuffer(nativeGetCurrentTaskId());
  }

  private static native int nativeGetCurrentTaskType();

  private static native ByteBuffer nativeGetCurrentTaskId();

  private static native ByteBuffer nativeGetCurrentJobId();

  private static native ByteBuffer nativeGetCurrentWorkerId();

  private static native ByteBuffer nativeGetCurrentActorId();
}
